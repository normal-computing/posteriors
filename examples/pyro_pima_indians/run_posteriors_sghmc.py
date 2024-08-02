from time import time
import torch
import os
import pickle
import posteriors
from tqdm import tqdm

from examples.pyro_pima_indians.model import load_data, load_model

repeats = 10
batchsize = None

save_dir = "examples/pyro_pima_indians/results/"
save_path = os.path.join(save_dir, f"posteriors_sghmc_{batchsize}.pkl")

N_warmup_posteriors = 2000
N_samples_posteriors = 5000
thinning_posteriors = 5
N_posteriors = N_warmup_posteriors + N_samples_posteriors * thinning_posteriors

posteriors_sghmc_lr = 1e-2
posteriors_sghmc_alpha = 1.0
comp = False

X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

model, log_posterior_normalized = load_model(num_data)

samples = torch.zeros((repeats, N_samples_posteriors, dim))
log_posts = torch.zeros((repeats, N_posteriors))
times = torch.zeros(repeats)


for r in range(repeats):
    initial_params = torch.zeros(dim)

    # Fit with posteriors.sgmcmc
    transform = posteriors.sgmcmc.sghmc.build(
        log_posterior_normalized,
        lr=1e-2,
        temperature=1 / num_data,
        alpha=1.0,
    )

    state = transform.init(initial_params, momenta=0.0)

    all_params = torch.zeros(N_posteriors + 1, dim)
    all_params[0] = initial_params
    log_posts_r = torch.zeros(N_posteriors)

    update = torch.compile(transform.update) if comp else transform.update

    if batchsize is None:
        batches = [(X_all, y_all)] * N_posteriors
    else:
        batch_inds = [
            torch.randint(num_data, (batchsize,)) for _ in range(N_posteriors)
        ]
        batches = [(X_all[bi], y_all[bi]) for bi in batch_inds]

    _ = update(state, batches[0])

    posteriors_sghmc_time = time()

    for i in tqdm(range(N_posteriors)):
        state = update(state, batches[i])
        all_params[i] = state.params
        log_posts_r[i] = state.log_posterior

    posteriors_sghmc_time = time() - posteriors_sghmc_time

    all_params = all_params[N_warmup_posteriors + 1 :: thinning_posteriors].detach()

    samples[r] = all_params
    log_posts[r] = log_posts_r
    times[r] = posteriors_sghmc_time


save_dict = {
    "samples": samples.numpy(),
    "log_posts": log_posts.numpy(),
    "times": times.numpy(),
}


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)
