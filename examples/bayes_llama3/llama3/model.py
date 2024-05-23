from typing import List, Optional
from tqdm import tqdm

import posteriors
from transformers import AutoModelForCausalLM
from safetensors.torch import load_model
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

PRIOR_SD = 1e3


def log_posterior(model_func, num_data, vocab_size, ignore_first):
    def fn_call(params, inputs):
        outputs = model_func(params, **inputs)

        pred_logits = outputs.logits[:, ignore_first - 1 : -1, :].contiguous()
        labels = inputs["input_ids"][:, ignore_first:].contiguous()
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
        alpha: float = 1e-1,
        beta: float = 0.0,
        momenta: float = 0.0,
        ignore_first_tokens: int = 250,
        set_temperature: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.momenta = momenta
        self.set_temperature = set_temperature
        self.ignore_first_tokens = ignore_first_tokens
        if max_seq_len:
            assert ignore_first_tokens < max_seq_len
            max_seq_len = max_seq_len - ignore_first_tokens
            self.num_data = num_data * max_seq_len

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            pretrained_weights_folder, torch_dtype=torch.float16, device_map="auto"
        )
        self.num_decoder_layers = len(self.model.model.layers)

        self.vocab_size = self.model.config.vocab_size

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

    def training_step(self, batch):
        inputs = {key: val.to(self.device) for key, val in batch.items()}
        self.state = self.transform.update(self.state, inputs)
        self.log("loss", self.state.aux, prog_bar=True)

    def configure_optimizers(self):
        sub_params, sub_param_to_log_posterior = (
            posteriors.extract_requires_grad_and_func(
                dict(self.model.named_parameters()),
                log_posterior(
                    self.functional_model,
                    self.num_data,
                    self.vocab_size,
                    self.ignore_first_tokens,
                ),
            )
        )

        self.transform = posteriors.sgmcmc.sghmc.build(
            log_posterior=sub_param_to_log_posterior,
            temperature=1 / self.num_data if self.set_temperature else 0,
            lr=self.lr,
            alpha=self.alpha,
            beta=self.beta,
            momenta=self.momenta,
        )
        self.state = self.transform.init(sub_params)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = {"bayesian_layer": self.state}
