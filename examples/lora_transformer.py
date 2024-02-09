import regex as re
import numpy as np
from itertools import groupby
from functools import partial
from optree import tree_map, tree_reduce
import lightning as L
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from ml_collections.config_dict import FrozenConfigDict

import uqlib
from uqlib import model_to_function


class TransformerModule(L.LightningModule):
    def __init__(self, config: FrozenConfigDict):
        super().__init__()
        self.automatic_optimization = False

        self.pretrained_model_name_or_path = config.pretrained_model_name_or_path
        self.prior_sd = config.prior_sd

        self.target_modules = config.lora_config.target_modules
        self.r = config.lora_config.r
        self.alpha = config.lora_config.alpha
        self.dropout = config.lora_config.dropout

        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path)
        # only adapt W_q, W_v, W_o
        # regex may not work for all models
        modules = [
            re.sub("^(model\\.)*|(\\.weight)*$", "", name)
            for name, _ in model.named_parameters()
            if any(sub in name for sub in ["self_attn.q", "self_attn.v", "self_attn.o"])
        ]
        # only adapt last layer
        if self.target_modules == "last_layer":
            modules = [
                (
                    name,
                    np.array(
                        [int(sub) for sub in name.split(".") if sub.isdigit()]
                    ).item(),
                )
                for name in modules
            ]
            modules = [
                [name for name, layer in list(group)]
                for _, group in groupby(
                    sorted(modules, key=lambda x: x[-1]), key=lambda x: x[-1]
                )
            ][-1]

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

    @staticmethod
    def univariate_normal_log_prob(x, mean, sd):
        return -0.5 * ((x - mean) / sd) ** 2

    def normal_log_prior(self, p) -> float:
        per_group_vals = tree_map(
            lambda p: self.univariate_normal_log_prob(p, 0, self.prior_sd).sum(), p
        )
        return tree_reduce(torch.add, per_group_vals)

    def param_to_log_posterior(self, p, batch, num_data) -> torch.tensor:
        output = self.model_func(p, labels=batch["input_ids"], **batch)
        return (-output.loss) + self.normal_log_prior(p) / num_data, output

    def on_train_start(self) -> None:
        param_to_log_posterior = partial(
            self.param_to_log_posterior,
            num_data=len(self.trainer.train_dataloader.dataset),
        )

        (
            self.sub_params,
            self.sub_param_to_log_posterior,
        ) = uqlib.extract_requires_grad_and_func(
            dict(self.model.named_parameters()), param_to_log_posterior
        )
        self.opt = AdamW(self.sub_params.values(), lr=1e-5, maximize=True)

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        self.opt.zero_grad()

        log_post, out = self.sub_param_to_log_posterior(self.sub_params, batch)
        log_post.backward()

        self.log("log_post", log_post.item())
        self.opt.step()

        return torch.tensor(log_post.item())
