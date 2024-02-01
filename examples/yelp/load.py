from functools import partial
from datasets import load_dataset
from optree import tree_map, tree_reduce
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from uqlib import model_to_function


# From https://huggingface.co/docs/transformers/training#train-in-native-pytorch


def load_dataloaders(small=False, batch_size=8):
    dataset = load_dataset("yelp_review_full")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if small:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def load_model(
    prior_sd=1.0,
    num_data=None,
    per_sample=False,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    model_func = model_to_function(model)

    def categorical_log_likelihood(labels, logits):
        return Categorical(logits=logits, validate_args=False).log_prob(labels)

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
            categorical_log_likelihood(batch["labels"], output.logits)
        ) + normal_log_prior(p) / num_data, output

    if per_sample:
        param_to_log_posterior = param_to_log_posterior_per_sample
    else:

        def param_to_log_posterior(p, batch, num_data) -> float:
            log_probs, aux = param_to_log_posterior_per_sample(p, batch, num_data)
            return log_probs.mean(), aux

    if num_data is not None:
        param_to_log_posterior = partial(param_to_log_posterior, num_data=num_data)

    return model, param_to_log_posterior
