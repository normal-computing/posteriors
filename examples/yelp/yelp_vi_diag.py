import torch
from torch.distributions import Normal, Categorical
from torchopt import adamw
from tqdm.auto import tqdm
from optree import tree_map

import uqlib

from examples.yelp.load import load_dataloaders, load_model

train_dataloader, eval_dataloader = load_dataloaders(small=True)
model = load_model()

# Turn off Dropout
model.eval()

num_epochs = 3
num_data = len(train_dataloader.dataset)
num_training_steps = num_epochs * len(train_dataloader)


def categorical_log_likelihood(labels, logits):
    return Categorical(logits=logits, validate_args=False).log_prob(labels)


prior_sd = 1


def normal_log_prior(p: dict):
    return torch.stack(
        [
            Normal(0, prior_sd, validate_args=False).log_prob(ptemp).sum()
            for ptemp in p.values()
        ]
    ).sum()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


model_func = uqlib.model_to_function(model)


def param_to_log_posterior(p, batch):
    return (
        categorical_log_likelihood(batch["labels"], model_func(p, **batch).logits)
        + normal_log_prior(p) / num_data
    ).mean()


init_mean = dict(model.named_parameters())
init_log_sds = tree_map(
    lambda x: (torch.zeros_like(x) - 2.0).requires_grad_(True), init_mean
)

optimizer = adamw(lr=5e-5)

vi_state = uqlib.vi.diag.init(init_mean, optimizer=optimizer, init_log_sds=init_log_sds)

progress_bar = tqdm(range(num_training_steps))

nelbos = []

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        vi_state = uqlib.vi.diag.update(
            vi_state, param_to_log_posterior, batch, optimizer, 1
        )
        nelbos.append(vi_state.nelbo)
        progress_bar.update(1)


# # Alternative implementation that updates mu and log_sigma directly without using the
# # uqlib init+update API

# from torch.optim import AdamW
# from transformers import get_scheduler


# mu = dict(model.named_parameters())
# log_sigma = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), mu)

# vi_params_tensors = list(mu.values()) + list(log_sigma.values())

# vi_optimizer = AdamW(vi_params_tensors, lr=5e-5)
# vi_lr_scheduler = get_scheduler(
#     name="linear",
#     optimizer=vi_optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )

# progress_bar = tqdm(range(num_training_steps))

# nelbos = []

# # model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         vi_optimizer.zero_grad()

#         sigma = tree_map(torch.exp, log_sigma)

#         nelbo = uqlib.vi.diag.nelbo(
#             mu,
#             sigma,
#             param_to_log_posterior,
#             batch,
#         )

#         nelbo.backward()
#         nelbos.append(nelbo.item())

#         vi_optimizer.step()
#         vi_lr_scheduler.step()
#         progress_bar.update(1)
