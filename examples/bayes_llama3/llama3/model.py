from typing import List, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_model
from torch import func
import torchopt
import posteriors
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

# from modules.bayesllama import BayesLlamaForCausalLM

PRIOR_SD = 1e3


def log_posterior(model_func, num_data, vocab_size):
    def fn_call(params, inputs):
        outputs = model_func(params, **inputs)
        pred_logits = outputs.logits[:, :-1].contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
        pred_logits = pred_logits.view(-1, vocab_size)
        labels = labels.view(-1)

        loss = F.cross_entropy(pred_logits, labels)

        log_post = (
            -loss
            + posteriors.diag_normal_log_prob(params, sd_diag=PRIOR_SD, normalize=False)
            / num_data
        )
        return log_post, loss

    return fn_call


class BayesLlama(pl.LightningModule):
    def __init__(
        self,
        num_data: int,
        pretrained_weights_folder: str = "Meta-Llama-3-8B",
        lr: float = 1e-6,
    ):
        super().__init__()

        self.lr = lr
        self.num_data = num_data

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            pretrained_weights_folder, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights_folder)
        self.num_decoder_layers = len(self.model.model.layers)

        self.vocab_size = self.model.config.vocab_size
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.freeze_weights()
        self.functional_model = posteriors.model_to_function(self.model)

    def freeze_weights(self):
        for name, param in self.model.named_parameters():
            # Freeze everything but the last decoder layer
            if f".{self.num_decoder_layers - 1}." not in name:
                param.requires_grad = False

    def load_weights(self, weights_path: List[str]):
        print("Loading weights now")
        for path in tqdm(weights_path):
            load_model(self.model, path, strict=False)

    def batch_setup(self, batch):
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(
            self.device
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def training_step(self, batch):
        inputs = self.batch_setup(batch)
        self.state = self.transform.update(self.state, inputs)
        self.log("loss", self.state.aux, prog_bar=True)
        # TODO: Add logging

    def configure_optimizers(self):
        sub_params, sub_param_to_log_posterior = (
            posteriors.extract_requires_grad_and_func(
                dict(self.model.named_parameters()),
                log_posterior(self.functional_model, self.num_data, self.vocab_size),
            )
        )

        self.transform = posteriors.sgmcmc.sghmc.build(
            log_posterior=sub_param_to_log_posterior,
            temperature=1 / self.num_data,
            lr=self.lr,
        )
        self.state = self.transform.init(sub_params)

    # TODO: Implement save checkpoint to only save the last layer, we don't need all of these weights
    # def on_save_checkpoint(self, checkpoint: dict) -> None:
    # checkpoint["state_dict"] = {"bayesian_layer": }

    # @torch.no_grad()
    # def generate(self, batch, max_length=100, use_inversion_tokens=True):
    #     inputs, _ = self.batch_setup(
    #         batch["question"], batch["answer"], use_inversion_tokens
    #     )

    #     seq_out = []
    #     for _ in range(max_length):
    #         outputs = self.model(**inputs)

    #         if "attention_mask" in inputs:
    #             del inputs["attention_mask"]

    #         next_token = outputs["logits"][:, -1].argmax(-1).unsqueeze(-1)
    #         inputs["past_key_values"] = outputs["past_key_values"]
    #         inputs["inputs_embeds"] = self.model.model.embed_tokens(next_token)
    #         seq_out.append(next_token)

    #     seq_out = torch.cat(seq_out, -1)
    #     return self.tokenizer.batch_decode(seq_out, skip_special_tokens=True)
