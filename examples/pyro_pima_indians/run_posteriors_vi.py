from time import time
import torch
import os
import pickle
import posteriors
from tqdm import tqdm
import torchopt

from examples.pyro_pima_indians.model import load_data, load_model

repeats = 10
batchsize = None

save_dir = "examples/pyro_pima_indians/results/"
save_path = os.path.join(save_dir, f"posteriors_vi_{batchsize}.pkl")

steps_vi = 8000
N_samps_vi_save = 5000

comp = False

X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

model, log_posterior_normalized = load_model(num_data)

samples = torch.zeros((repeats, N_samps_vi_save, dim))
nelbos = torch.zeros((repeats, steps_vi))
times = torch.zeros(repeats)


for r in range(repeats):
    initial_params = torch.zeros(dim)

    # Fit with posteriors.vi
    transform = posteriors.vi.diag.build(
        log_posterior_normalized, torchopt.adam(), temperature=1 / num_data
    )

    state = transform.init(initial_params)

    nelbos_r = torch.zeros(steps_vi)

    if batchsize is None:
        batches = [(X_all, y_all)] * steps_vi
    else:
        batch_inds = [torch.randint(num_data, (batchsize,)) for _ in range(steps_vi)]
        batches = [(X_all[bi], y_all[bi]) for bi in batch_inds]

    posteriors_vi_time = time()

    update = torch.compile(transform.update) if comp else transform.update

    for i in tqdm(range(steps_vi)):
        state, _ = update(state, batches[i])
        nelbos_r[i] = state.nelbo

    posteriors_vi_time = time() - posteriors_vi_time

    samples[r] = posteriors.vi.diag.sample(state, (N_samps_vi_save,))
    nelbos[r] = nelbos_r
    times[r] = posteriors_vi_time


save_dict = {
    "samples": samples.numpy(),
    "nelbos": nelbos.numpy(),
    "times": times.numpy(),
}


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)
