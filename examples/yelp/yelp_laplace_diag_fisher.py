import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import uqlib

from examples.yelp.load import load_dataloaders, load_model

# Load data
train_dataloader, eval_dataloader = load_dataloaders(small=True)
num_data = len(train_dataloader.dataset)

# Load model (with Gaussian prior)
model, param_to_log_posterior = load_model(num_data=num_data)

# Turn off Dropout
model.eval()

# Move to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Train (as usual, using native PyTorch) for MAP
optimizer = AdamW(model.parameters(), lr=5e-5, maximize=True)

num_epochs = 3
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

        log_post = param_to_log_posterior(dict(model.named_parameters()), batch)

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


# # Diagonal Hessian covariance matrix
# progress_bar_2 = tqdm(range(len(train_dataloader)))

# laplace_state = uqlib.laplace.diag_hessian.init(dict(model.named_parameters()))

# for batch in train_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     laplace_state = uqlib.laplace.diag_hessian.update(
#         laplace_state, param_to_log_posterior, batch
#     )
#     progress_bar_2.update(1)
