from functools import partial
import pickle
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from uqlib import (
    model_to_function,
    diag_normal_log_prob,
    extract_requires_grad_and_func,
    tree_insert,
)


# From https://huggingface.co/docs/transformers/training#train-in-native-pytorch


def load_dataloaders(small=False, batch_size=8, test_batch_size=None, seed=None):
    if test_batch_size is None:
        test_batch_size = batch_size

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

    if seed is not None:
        train_dataset = train_dataset.shuffle(seed=seed)
        eval_dataset = eval_dataset.shuffle(seed=seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=test_batch_size)

    return train_dataloader, eval_dataloader


def load_spanish_dataloader(small=False, batch_size=8):
    dataset = load_dataset("beltrewilton/punta-cana-spanish-reviews")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["review_text"], padding="max_length", truncation=True)

    dataset = dataset.map(lambda x: {"labels": x["rating"] - 1})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["hotel_name", "location", "wrote", "title", "review_text", "rating"]
    )
    tokenized_datasets.set_format("torch")

    if small:
        dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    else:
        dataset = tokenized_datasets["train"]

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader


def load_model(
    prior_sd=1.0,
    num_data=None,
    params_dir=None,
    last_layer_params_dir=None,
    last_layer=False,
    device="cpu",
):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    model.to(device)

    model_func = model_to_function(model)

    def categorical_log_likelihood(labels, logits):
        return Categorical(logits=logits, validate_args=False).log_prob(labels)

    def param_to_log_posterior(p, batch, num_data):
        output = model_func(p, **batch)
        return (
            categorical_log_likelihood(batch["labels"], output.logits).mean()
        ) + diag_normal_log_prob(
            p, sd_diag=prior_sd, normalize=False
        ) / num_data, output

    if num_data is not None:
        param_to_log_posterior = partial(param_to_log_posterior, num_data=num_data)

    if params_dir is not None:
        state = pickle.load(open(params_dir, "rb"))
        params = tree_insert(
            lambda x: x.numel() == 0,
            state.params,
            dict(model.named_parameters()),
        )
        if last_layer_params_dir is not None:
            last_layer_state = pickle.load(open(last_layer_params_dir, "rb"))
            params = tree_insert(
                lambda x: x.numel() == 0,
                last_layer_state.params,
                params,
            )

        model.load_state_dict(params)

    if last_layer:
        for name, param in model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        # Extract only the parameters to be trained
        sub_params, sub_param_to_log_posterior = extract_requires_grad_and_func(
            dict(model.named_parameters()), param_to_log_posterior
        )

        return model, sub_param_to_log_posterior, sub_params

    return model, param_to_log_posterior, dict(model.named_parameters())
