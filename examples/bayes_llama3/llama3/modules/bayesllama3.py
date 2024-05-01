from typing import List
from tqdm import tqdm

from posteriors import model_to_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_model
import pytorch_lightning as pl
import torch.nn as nn
import torch


class BayesLlama(pl.LightningModule):
    def __init__(
        self,
        pretrained_weights_folder: str = "Meta-Llama-3-8B",
        lr: float = 1e-6,
        weight_decay: float = 0.1,
        reg_strength: float = 1,
    ):
        super().__init__()

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            pretrained_weights_folder, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights_folder)

        self.freeze_weights()
        self.model_func = model_to_function(self.model)

    def freeze_weights(self):
        for name, param in self.model.named_parameters():
            if "lm_head" not in name:
                param.requires_grad_ = False

    def load_weights(self, weights_path: List[str]):
        print("Loading weights now")
        for path in tqdm(weights_path):
            load_model(self.model, path, strict=False)

    def batch_setup(self, batch):
        return self.tokenizer(batch["model"], return_tensors="pt", padding=True).to(
            self.device
        )

    def training_step(self, batch):
        # inputs = self.batch_setup(batch)
        # question, answer = batch["question"], batch["answer"]
        # self.model_func()

        # self.state = self.transform.update(self.state, batch)
        # TODO: Add logging
        pass

    def configure_optimizers(self):
        pass
        # self.transform = config.method.build(
        #     sub_param_to_log_posterior, **config.config_args
        # )
        # self.state = self.transform.init(sub_params)

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
