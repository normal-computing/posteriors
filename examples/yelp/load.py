from functools import partial
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal, Categorical
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


def load_model(prior_sd=1, num_data=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    model_func = model_to_function(model)

    def categorical_log_likelihood(labels, logits):
        return Categorical(logits=logits, validate_args=False).log_prob(labels)

    def normal_log_prior(p: dict) -> float:
        return torch.stack(
            [
                Normal(0, prior_sd, validate_args=False).log_prob(ptemp).sum()
                for ptemp in p.values()
            ]
        ).sum()

    def param_to_log_posterior(p, batch, num_data) -> torch.tensor:
        return (
            categorical_log_likelihood(batch["labels"], model_func(p, **batch).logits)
            + normal_log_prior(p) / num_data
        ).mean()

    if num_data is not None:
        param_to_log_posterior = partial(param_to_log_posterior, num_data=num_data)

    return model, param_to_log_posterior
