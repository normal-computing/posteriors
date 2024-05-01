import optree
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from transformers import AutoTokenizer

import posteriors
from examples.bayes_llama3.model.bayesllama3 import BayesLlamaForCausalLM

from posteriors import model_to_function

# From https://huggingface.co/docs/transformers/training#train-in-native-pytorch


def load_dataloaders(small=False, batch_size=8, test_batch_size=None, seed=None):
    if test_batch_size is None:
        test_batch_size = batch_size

    dataset = load_dataset("yelp_review_full")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

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


def load_model(
    bayes_config,
    prior_sd=1.0,
    device="cpu",
):
    model = BayesLlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", bayes_config=bayes_config
    )
    model.to(device)
    model_func = model_to_function(model)

    for name, param in model.named_parameters():
        if "bayesian_ensemble" not in name:
            param.requires_grad = False

    def logits_to_log_lik(logits, labels, keep_mask):
        # pred_seq_len = seq_length - ignore_first
        logits_start = bayes_config.ignore_first
        logits = logits[
            :, (logits_start - 1) : -1, :
        ].contiguous()  # (batch_size, pred_seq_len, vocab_size)
        labels = labels[:, logits_start:].contiguous()  # (batch_size, pred_seq_len)
        keep_mask = keep_mask[
            :, logits_start:
        ].contiguous()  # (batch_size, pred_seq_len)
        log_lik_all = Categorical(logits=logits, validate_args=False).log_prob(labels)
        log_lik_all *= keep_mask
        # sum over sequence, average over batch to give unbiased estimate of single sequence log likelihood
        # (within sequence log likelihoods are not independent)
        log_lik = log_lik_all.sum(1).mean()
        return log_lik  # rescale to full log likelihood

    # Full parameter set log_likelihood
    def param_to_log_likelihood(params, batch):
        output = model_func(params, labels=batch["input_ids"], **batch)
        log_lik = logits_to_log_lik(
            output.logits, batch["input_ids"], batch["attention_mask"]
        )
        return log_lik, output

    # Extract only the LoRA parameters that require gradients
    sub_params, sub_param_to_log_likelihood = posteriors.extract_requires_grad_and_func(
        dict(model.named_parameters()), param_to_log_likelihood
    )

    # Generic (unnormalised) Normal log prior
    def normal_log_prior(params, prior_mean, prior_sd) -> float:
        per_group_vals = optree.tree_map(
            lambda p, m, sd: (-0.5 * ((p - m) / sd) ** 2).sum(),
            params,
            prior_mean,
            prior_sd,
        )
        return optree.tree_reduce(torch.add, per_group_vals)

    def sub_param_to_log_posterior(p, batch, prior_mean, num_sequences):
        log_lik, output = sub_param_to_log_likelihood(p, batch)
        log_prior = normal_log_prior(p, prior_mean, prior_sd)
        log_post = log_lik + log_prior / num_sequences
        return log_post, output

    return model, sub_param_to_log_posterior, sub_params
