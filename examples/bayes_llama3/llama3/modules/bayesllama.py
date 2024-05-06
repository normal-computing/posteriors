from typing import Optional, Tuple, Union, List
import copy

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
import torch.nn.functional as F
import regex as re


logger = logging.get_logger(__name__)


class BayesLlamaModel(LlamaModel):
    def __init__(self, config, bayes_config):
        super().__init__(config)

        self.bayesian_layers = nn.ModuleList(
            [copy.deepcopy(self.layers[-1]) for _ in range(bayes_config["n_ensemble"])]
        )
        for layer in self.bayesian_layers:
            layer.self_attn.layer_idx = 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        ensemble_past_key_values: Optional[List[List[torch.FloatTensor]]] = None,
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

            if ensemble_past_key_values is None:
                ensemble_past_key_values = [
                    None for _ in range(len(self.bayesian_layers))
                ]
            if not isinstance(ensemble_past_key_values[0], StaticCache):
                ensemble_past_key_values = [
                    DynamicCache.from_legacy_cache(pkv)
                    for pkv in ensemble_past_key_values
                ]
                assert (
                    ensemble_past_key_values[0].get_seq_length() == past_seen_tokens
                ), "Ensemble KV cache and decoder cache is out of sync"

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

        for idx, decoder_layer in enumerate(self.layers[:-1]):
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

        bayesian_layer_outputs = []
        ensemble_next_decoder_cache = []
        for idx, layer in enumerate(self.bayesian_layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=(
                    ensemble_past_key_values[idx]
                    if ensemble_past_key_values is not None
                    else None
                ),
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            bayesian_layer_outputs.append(layer_outputs)

            if use_cache:
                ensemble_next_decoder_cache.append(
                    layer_outputs[2 if output_attentions else 1]
                )

        if output_attentions:  # Output attentions from all bayesian layers
            all_self_attns += (
                [layer_outputs[1] for layer_outputs in bayesian_layer_outputs],
            )

        # stack hidden states from bayesian layers
        hidden_states = torch.stack(
            [bayesian_layer_outputs[i][0] for i in range(len(bayesian_layer_outputs))]
        )
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        ensemble_next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
            ensemble_next_cache = [
                c.to_legacy_cache() if isinstance(c, Cache) else c
                for c in ensemble_next_decoder_cache
            ]
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    ensemble_next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
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

    def __init__(self, config, bayes_config):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        self.model = BayesLlamaModel(config, bayes_config=bayes_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def load_bayesian_layers(self, layer_state_dicts):
        assert len(layer_state_dicts) == len(self.model.bayesian_layers)

        for state_dict, bayesian_layer in zip(
            layer_state_dicts, self.model.bayesian_layers
        ):
            bayesian_layer.load_state_dict(state_dict)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        ensemble_past_key_values: Optional[List[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
            ensemble_past_key_values=ensemble_past_key_values,
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

        logits = logits.float()
        # logits are size (n_ensemble, batch_size, seq_len, vocab_size)
        # mean over n_ensemble to get (batch_size, seq_len, vocab_size)
        mean_logits = logits.mean(dim=0)

        if not return_dict:
            return ((mean_logits, logits),) + outputs[1:]

        return CausalLMOutputWithPast(
            logits=(mean_logits, logits),  # return logits from all bayesian layers
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
