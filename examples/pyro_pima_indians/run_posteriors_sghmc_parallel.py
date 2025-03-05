from time import time
import torch
import os
import pickle
import posteriors
from tqdm import tqdm

from examples.pyro_pima_indians.model import load_data, load_model

repeats = 10000
batchsize = 32

save_sample_size = 5000
save_repeats = 10

save_dir = "examples/pyro_pima_indians/results/"
save_path = os.path.join(save_dir, f"posteriors_sghmc_parallel_{batchsize}.pkl")

N_steps_posteriors = 2000

posteriors_sghmc_lr = 1e-2
posteriors_sghmc_alpha = 1.0
comp = False

X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

model, log_posterior_normalized = load_model(num_data)

samples = torch.zeros((repeats, dim))
times = torch.zeros(repeats)

initial_params = torch.zeros(dim)

transform = posteriors.sgmcmc.sghmc.build(
    log_posterior_normalized,
    lr=1e-2,
    temperature=1 / num_data,
    alpha=1.0,
)


def run_chain(batches):
    state = transform.init(initial_params, momenta=0.0)

    for i in range(len(batches)):
        if batchsize is None:
            batch = (X_all, y_all)
        else:
            batch = batches[i]

        state, _ = transform.update(state, batch)

    return state


run_chain_c = torch.compile(run_chain) if comp else run_chain

# Compile
_ = run_chain_c([(X_all[:batchsize], y_all[:batchsize])])


for r in tqdm(range(repeats)):
    batch_inds = [
        torch.randint(num_data, (batchsize,)) for _ in range(N_steps_posteriors)
    ]
    batches = [(X_all[bi], y_all[bi]) for bi in batch_inds]

    posteriors_psghmc_time = time()

    state = run_chain_c(batches)

    posteriors_psghmc_time = time() - posteriors_psghmc_time

    samples[r] = state.params
    times[r] = posteriors_psghmc_time


bootstrap_inds = torch.randint(repeats, (save_repeats, save_sample_size))
save_samples = samples[bootstrap_inds]


save_dict = {
    "samples": save_samples.numpy(),
    "times": times.numpy(),
}


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)
