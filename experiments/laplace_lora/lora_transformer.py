from typing import Tuple
from itertools import groupby
from optree import tree_map, tree_reduce
import lightning as L
import torch
from torch.optim import AdamW
from torchmetrics import Accuracy
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from ml_collections.config_dict import FrozenConfigDict

import uqlib
from uqlib import model_to_function


class BayesTransformerModule(L.LightningModule):
    def __init__(self, config: FrozenConfigDict):
        super().__init__()
        self.automatic_optimization = False

        self.pretrained_model_name_or_path = config.pretrained_model_name_or_path
        self.lr = config.lr

        # These need to updated with the correct values before calling trainer.fit
        self.prior_mean = None
        self.prior_sd = None
        self.num_data = None

        self.target_modules = config.lora_config.target_modules
        self.r = config.lora_config.r
        self.alpha = config.lora_config.alpha
        self.dropout = config.lora_config.dropout

        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path)
        # only adapt W_q, W_v, W_o
        # regex may not work for all models
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=model.config.vocab_size
        )

        WEIGHTS_TO_LORA = ["q_proj", "v_proj", "o_proj"]

        modules = list(model.model.layers.named_parameters())
        # Get layer index, name for layers to adapt
        module_names_with_layer = [
            (name.split(".")[0], f"layers.{name.strip('.weight')}")
            for name, param in modules
            if any(
                sub in name
                for sub in [
                    "self_attn.{sub}".format(sub=sub) for sub in WEIGHTS_TO_LORA
                ]
            )
        ]

        # Subset of layers to adapt
        if self.target_modules == "last_layer":
            modules = [
                [layer for name, layer in list(group)]
                for _, group in groupby(module_names_with_layer, key=lambda x: x[0])
            ][-1]
        else:
            modules = [name for layer, name in module_names_with_layer]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=modules,
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
        )

        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.model_func = model_to_function(self.model)

        (
            self.sub_params,
            self.sub_param_to_log_likelihood,
        ) = uqlib.extract_requires_grad_and_func(
            dict(self.model.named_parameters()), self.param_to_log_likelihood
        )

    @staticmethod
    def univariate_normal_log_prob(x, mean, sd):
        return -0.5 * ((x - mean) / sd) ** 2

    def normal_log_prior(self, params) -> float:
        per_group_vals = tree_map(
            lambda p, m, sd: self.univariate_normal_log_prob(p, m, sd).sum(),
            params,
            self.prior_mean,
            self.prior_sd,
        )
        return tree_reduce(torch.add, per_group_vals)

    def param_to_log_likelihood(
        self, p, batch
    ) -> Tuple[torch.tensor, uqlib.types.TensorTree]:
        output = self.model_func(p, labels=batch["input_ids"], **batch)
        return -output.loss, output

    def sub_param_to_log_posterior(
        self, p, batch
    ) -> Tuple[torch.tensor, uqlib.types.TensorTree]:
        log_lik, output = self.sub_param_to_log_likelihood(p, batch)
        log_prior = self.normal_log_prior(p)
        return log_lik + log_prior / self.num_data, output

    def configure_optimizers(self):
        self.opt = AdamW(self.sub_params.values(), lr=self.lr, maximize=True)
        self.sub_params = tree_map(lambda x: x.to(self.device), self.sub_params)
        self.prior_mean = tree_map(lambda x: x.to(self.device), self.prior_mean)
        self.prior_sd = tree_map(lambda x: x.to(self.device), self.prior_sd)

    def training_step(self, batch, batch_idx):
        self.opt.zero_grad()

        log_post, out = self.sub_param_to_log_posterior(self.sub_params, batch)
        log_post.backward()

        self.log("log_post", log_post.item())
        self.log("loss", out.loss)
        self.opt.step()

        return log_post

    def validation_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        targets = inputs[:, 1:].clone().detach()

        with torch.no_grad():
            output = self.model(
                input_ids=inputs,
                labels=inputs,
            )
        logits = output.logits[:, :-1, :]
        preds = logits.argmax(-1)

        self.val_accuracy.update(preds, targets)
        self.log(
            "val_loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return output.loss

    def on_validation_epoch_end(
        self,
    ):
        self.log(
            "val_accuracy",
            self.val_accuracy.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        print(f"Validation Accuracy: {self.val_accuracy.compute()}")
