from time import time
import pyro
import torch
import os
import pickle

from examples.pyro_pima_indians.model import load_data, load_model

save_dir = "examples/pyro_pima_indians/results/"
save_path = os.path.join(save_dir, "pyro.pkl")

repeats = 10

N_warmup_pyro = 2000
N_samples_pyro = 5000


X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

model, log_posterior_normalized = load_model(num_data)

samples = torch.zeros((repeats, N_samples_pyro, dim))
log_posts = torch.zeros((repeats, N_samples_pyro))
times = torch.zeros(repeats)

for r in range(repeats):
    pyro_nuts_time = time()

    initial_params = torch.zeros(dim)

    nuts_kernel = pyro.infer.NUTS(model, adapt_step_size=True)
    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=N_samples_pyro,
        warmup_steps=N_warmup_pyro,
        initial_params={"w": initial_params},
    )
    mcmc.run((X_all, y_all))
    samps = mcmc.get_samples()["w"]

    pyro_nuts_time = time() - pyro_nuts_time

    log_posts_pyro = torch.tensor(
        [log_posterior_normalized(samp, (X_all, y_all))[0] for samp in samps]
    )

    samples[r] = samps
    log_posts[r] = log_posts_pyro
    times[r] = pyro_nuts_time


save_dict = {
    "samples": samples.numpy(),
    "log_posts": log_posts.numpy(),
    "times": times.numpy(),
}

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)
