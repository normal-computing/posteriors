import torch
from torch.distributions import Normal, Categorical
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import uqlib

from examples.yelp.load import load_dataloaders, load_model

train_dataloader, eval_dataloader = load_dataloaders(small=True)
model = load_model()

# Turn off Dropout
model.eval()


def categorical_log_likelihood(labels, logits):
    return Categorical(logits=logits).log_prob(labels)


prior_sd = 1


def normal_log_prior(p: dict):
    return torch.stack(
        [Normal(0, prior_sd).log_prob(ptemp).sum() for ptemp in p.values()]
    ).sum()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model_func = uqlib.model_to_function(model)


def param_to_log_posterior(p, batch):
    return (
        categorical_log_likelihood(batch["labels"], model_func(p, **batch).logits)
        + normal_log_prior(p) / num_data
    )


# Train (as usual using native PyTorch) for MAP
optimizer = AdamW(model.parameters(), lr=5e-5, maximize=True)

num_epochs = 3
num_data = len(train_dataloader.dataset)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


progress_bar = tqdm(range(num_training_steps))

log_posts = []

# model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        log_post = param_to_log_posterior(dict(model.named_parameters()), batch).mean()

        log_post.backward()
        log_posts.append(log_post.item())

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# Use uqlib for diagonal Fisher information covariance matrix
progress_bar_2 = tqdm(range(len(train_dataloader)))

laplace_state = uqlib.laplace.diag_fisher.init(dict(model.named_parameters()))

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    laplace_state = uqlib.laplace.diag_fisher.update(
        laplace_state, param_to_log_posterior, batch
    )
    progress_bar_2.update(1)
