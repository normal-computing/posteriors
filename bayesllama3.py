import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple, Union
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging
import einops

logger = logging.get_logger(__name__)

class BayesLlamaModel(LlamaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens=0)

        # embed positions
        hidden_states = inputs_embeds

        for decoder_layer in self.layers[:-1]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]

        #Define pass through ensembled last layer
        def bayes_layer(ensemble_params, hidden_states, causal_mask, position_ids,):
            final_decoder = self.layers[-1]
            final_decoder.self_attn.o_proj.weight = torch.nn.Parameter(ensemble_params.to(self.device))

            layer_outputs = final_decoder(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids
                )
            return layer_outputs[0]

        hidden_states = torch.vmap(bayes_layer, in_dims=(0, None, None,None))(self.bayes_ensemble, hidden_states, causal_mask, position_ids)

        return self.norm(hidden_states)
    

class BayesLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BayesLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.n_ensemble = 10
        self.bayes_ensemble = None

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        #Init bayes ensemble
        if self.n_ensemble>0 and self.model.bayes_ensemble is None:
            self.model.bayes_ensemble = torch.stack([copy.deepcopy(self.model.layers[-1].self_attn.o_proj.weight) for _ in range(self.n_ensemble)])

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        logits = self.lm_head(hidden_states)
        logits = einops.rearrange(logits, 'e b s v -> b e s v')

        return logits.float()