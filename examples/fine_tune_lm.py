import torch
from torch.distributions import Normal, Categorical
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.func import grad
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
from tqdm.auto import tqdm

import uqlib

# Replication/modification of
# https://huggingface.co/docs/transformers/training#train-in-native-pytorch

dataset = load_dataset("yelp_review_full")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=5
)

# Turn off Dropout
model.eval()

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_data = len(train_dataloader.dataset)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def categorical_log_likelihood(labels, logits):
    return Categorical(logits=logits).log_prob(labels).mean(dim=-1)


prior_sd = 1


def normal_log_prior(p: dict):
    return torch.stack(
        [Normal(0, prior_sd).log_prob(ptemp).sum() for ptemp in p.values()]
    ).sum()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


model_func = uqlib.model_to_function(model)


def param_to_neg_log_posterior(p, batch):
    return (
        -categorical_log_likelihood(batch["labels"], model_func(p, **batch).logits)
        - normal_log_prior(p) / num_data
    )


progress_bar = tqdm(range(num_training_steps))

losses = []

# model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        loss = param_to_neg_log_posterior(dict(model.named_parameters()), batch)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


params = dict(model.named_parameters())

# # Laplace with diagonal Hessian for approximate precision matrix
# diag_hess = uqlib.tree_map(lambda x: torch.zeros_like(x, requires_grad=False), params)

# for batch in train_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         batch_diag_hess = uqlib.diagonal_hessian(param_to_neg_log_posterior)(
#             params, batch
#         )
#     diag_hess = uqlib.tree_map(lambda x, y: x + y, diag_hess, batch_diag_hess)


# Laplace with diagonal empirical Fisher for approximate precision matrix
diag_fish = uqlib.tree_map(lambda x: torch.zeros_like(x, requires_grad=False), params)

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        batch_diag_score = grad(param_to_neg_log_posterior)(params, batch)
    diag_fish = uqlib.tree_map(lambda x, y: x + y**2, diag_fish, batch_diag_score)


# VI
num_vi_epochs = num_epochs
num_vi_training_steps = num_vi_epochs * len(train_dataloader)


mu = params.copy()
log_sigma = uqlib.tree_map(lambda x: torch.zeros_like(x, requires_grad=True), params)

vi_params_tensors = list(mu.values()) + list(log_sigma.values())

vi_optimizer = AdamW(vi_params_tensors, lr=5e-5)
vi_lr_scheduler = get_scheduler(
    name="linear",
    optimizer=vi_optimizer,
    num_warmup_steps=0,
    num_training_steps=num_vi_training_steps,
)

progress_bar = tqdm(range(num_vi_training_steps))

nelbos = []

# model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        nelbo = uqlib.vi.diagonal_nelbo(
            param_to_neg_log_posterior, batch, mu, log_sigma
        )

        nelbo.backward()
        nelbos.append(nelbo.item())

        vi_optimizer.step()
        vi_lr_scheduler.step()
        vi_optimizer.zero_grad()
        progress_bar.update(1)
