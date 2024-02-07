from functools import partial
from itertools import groupby
import numpy as np
import regex as re
from datasets import load_dataset
from optree import tree_map, tree_reduce
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

from uqlib import model_to_function


# From https://huggingface.co/docs/transformers/training#train-in-native-pytorch


def load_dataloaders(small=False, batch_size=8):
    dataset = load_dataset("timdettmers/openassistant-guanaco")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", max_length=100, truncation=True
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    if small:
        train_dataset = train_dataset.shuffle(seed=42).select(range(100))
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(100))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def load_model(
    prior_sd=1.0,
    num_data=None,
    per_sample=False,
    target_modules=None,
    r=8,
    alpha=32,
    dropout=0.1,
    verbose=False,
):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
    )
    # only adapt W_q, W_v, W_o
    # regex may not work for all models
    modules = [
        re.sub("^(model\\.)*|(\\.weight)*$", "", name)
        for name, _ in model.named_parameters()
        if any(sub in name for sub in ["self_attn.q", "self_attn.v", "self_attn.o"])
    ]
    # only adapt last layer
    if target_modules == "last_layer":
        modules = [
            (
                name,
                np.array([int(sub) for sub in name.split(".") if sub.isdigit()]).item(),
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
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    model = get_peft_model(model, peft_config)
    if verbose:
        model.print_trainable_parameters()

    model_func = model_to_function(model)

    def categorical_log_likelihood(labels, logits):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def univariate_normal_log_prob(x, mean, sd):
        return -0.5 * ((x - mean) / sd) ** 2

    def normal_log_prior(p) -> float:
        per_group_vals = tree_map(
            lambda p: univariate_normal_log_prob(p, 0, prior_sd).sum(), p
        )
        return tree_reduce(torch.add, per_group_vals)

    def param_to_log_posterior_per_sample(p, batch, num_data) -> torch.tensor:
        output = model_func(p, **batch)

        return (
            categorical_log_likelihood(batch["input_ids"], output.logits)
        ) + normal_log_prior(p) / num_data, output

    if per_sample:
        param_to_log_posterior = param_to_log_posterior_per_sample
    else:

        def param_to_log_posterior(p, batch, num_data) -> float:
            log_probs, aux = param_to_log_posterior_per_sample(p, batch, num_data)
            return log_probs.mean(), aux

    if num_data is not None:
        param_to_log_posterior = partial(param_to_log_posterior, num_data=num_data)

    return model, param_to_log_posterior, modules
