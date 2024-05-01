from typing import List
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


def log_posterior(num_data):
    def fn_call(params, output, labels):
        log_post_val = (
            -F.cross_entropy(output, labels)
            + posteriors.diag_normal_log_prob(params) / num_data
        )
        return log_post_val

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

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            pretrained_weights_folder, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights_folder)
        self.num_decoder_layers = len(self.model.model.layers)

        self.vocab_size = self.model.config.vocab_size
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "left"

        self.log_posterior = log_posterior(num_data)
        self.num_data = num_data

        self.freeze_weights()
        self.params = dict(self.model.named_parameters())

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
        return inputs

    def training_step(self, batch):
        inputs = self.batch_setup(batch)
        input_ids = inputs["input_ids"]
        print(inputs)
        logits = func.functional_call(self.model, self.params, inputs)

        pred_logits = logits[:, self.inversion_token_seq_len : -1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        self.state = self.transform.update(self.state, pred_logits, labels)
        # TODO: Add logging

    def configure_optimizers(self):
        self.transform = posteriors.sgmcmc.sghmc.build(
            log_posterior=log_posterior,
            # temperature=1 / self.num_data,
            lr=self.lr,
        )
        self.state = self.transform.init(self.params)

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
