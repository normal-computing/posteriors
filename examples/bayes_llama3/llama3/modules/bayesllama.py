from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from torch.nn import CrossEntropyLoss, functional as F


logger = logging.get_logger(__name__)


class BayesLlamaModel(LlamaModel):
    def __init__(self, config, bayes_config):
        super().__init__(config)

        self.n_ensemble = bayes_config["n_ensemble"]
        self.ensemble_param_names = bayes_config["ensemble_param_names"]
        self.ensemble_param_layer = bayes_config["ensemble_param_layer"]

    def get_module_weights(self, module, param_name: str):
        attributes = param_name.split(".")
        for attr in attributes:
            module = getattr(module, attr)

        return module

    def load_ensemble_weights(
        self, layer_idx: int, param_names: list, ensemble_params: list[torch.Tensor]
    ):
        module = self.layers[layer_idx]

        for param_name, params in zip(param_names, ensemble_params):
            attributes = param_name.split(".")

            sub_module = module
            attr = None
            for attr in attributes:
                sub_module = getattr(sub_module, attr)
            setattr(sub_module, attr, torch.nn.Parameter(params.to(self.device)))

        return module

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    "cache_position is a required argument when using StaticCache."
                )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers[:-1]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Define pass through ensembled last layer
        def bayes_layer(
            ensemble_params,
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            use_cache,
            output_attentions,
            cache_position,
            all_self_attns,
        ):
            final_decoder = self.load_ensemble_weights(
                layer_idx=self.ensemble_param_layer,
                param_names=self.ensemble_param_names,
                ensemble_params=ensemble_params,
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    final_decoder.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = final_decoder(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache = next_decoder_cache.to_legacy_cache()

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            return (
                hidden_states,
                all_self_attns if output_attentions else torch.empty(0),
                next_decoder_cache if use_cache else torch.empty(0),
            )

        get_names = [
            "_".join(param_nm.split(".")) for param_nm in self.ensemble_param_names
        ]
        layer_outputs = torch.vmap(bayes_layer, in_dims=(0, None))(
            torch.stack(
                [
                    torch.stack(
                        [
                            getattr(self, f"bayesian_ensemble_{param_nm}_{en}")
                            for param_nm in get_names
                        ]
                    )
                    for en in range(self.n_ensemble)
                ]
            ),
            hidden_states,
            causal_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            all_self_attns=all_self_attns,
        )

        hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_attentions:
            all_self_attns = layer_outputs[1]

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = layer_outputs[2 if output_attentions else 1]
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BayesLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, bayes_config=None):
        super().__init__(config)
        self.model = BayesLlamaModel(config, bayes_config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.n_ensemble = bayes_config["n_ensemble"]
        self.ensemble_param_names = bayes_config["ensemble_param_names"]
        self.ensemble_param_layer = bayes_config["ensemble_param_layer"]
        self.bayes_ensemble_initialized = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.n_ensemble > 0 and not self.bayes_ensemble_initialized:
            for param_nm in self.ensemble_param_names:
                set_param_nm = "_".join(param_nm.split("."))
                for en in range(self.n_ensemble):
                    setattr(
                        self.model,
                        f"bayesian_ensemble_{set_param_nm}_{en}",
                        self.model.get_module_weights(
                            self.model.layers[self.ensemble_param_layer], param_nm
                        )
                        .clone()
                        .detach(),
                    )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.transpose(0, 1)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
