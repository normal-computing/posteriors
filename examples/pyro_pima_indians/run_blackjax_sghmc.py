from time import time
from jax import numpy as jnp, jit, random, grad
import os
import pickle
import torch
import blackjax
import numpy as np

from tqdm import tqdm

from examples.pyro_pima_indians.model import load_data, load_model, load_jax_model

save_dir = "examples/pyro_pima_indians/results/"
save_path = os.path.join(save_dir, "blackjax_sghmc.pkl")

repeats = 10

N_warmup_bj = 2000
N_samples_bj = 5000
thinning_bj = 5
N_bj = N_warmup_bj + N_samples_bj * thinning_bj


bj_stepsize = 1e-2
bj_alpha = 1.0
jit_bool = True


X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

X_all_jax = jnp.array(X_all.numpy())
y_all_jax = jnp.array(y_all.numpy())

log_posterior_normalized_torch = load_model(num_data)[1]
log_posterior_normalized = load_jax_model(num_data)

samples = jnp.zeros((repeats, N_samples_bj, dim))
log_posts = jnp.zeros((repeats, N_bj))
times = jnp.zeros(repeats)


jax_eval = log_posterior_normalized(jnp.ones(dim), (X_all_jax, y_all_jax))
torch_eval = log_posterior_normalized_torch(torch.ones(dim), (X_all, y_all))[0]
assert jnp.isclose(jax_eval, torch_eval.numpy())

rng_keys = random.split(random.PRNGKey(0), repeats)

for r in range(repeats):
    rng_key = rng_keys[r]

    bj_sghmc_time = time()

    bj_position = jnp.zeros(dim)
    bj_momentum = jnp.zeros_like(bj_position)

    bj_diffusion_step = blackjax.sgmcmc.diffusions.sghmc(alpha=bj_alpha)

    def bj_step(rng_key, position, momentum, batch):
        logdensity_grad = grad(log_posterior_normalized)(position, batch)
        return bj_diffusion_step(
            rng_key,
            position,
            momentum,
            logdensity_grad,
            bj_stepsize,
            temperature=1 / num_data,
        )

    bj_step_jit = jit(bj_step) if jit_bool else bj_step

    bj_all_params = jnp.zeros((N_bj + 1, dim)).at[0].set(bj_position)

    # compile the function
    _ = bj_step_jit(rng_key, bj_position, bj_momentum, (X_all_jax, y_all_jax))

    for i in tqdm(range(N_bj)):
        rng_key, subkey = random.split(rng_key)
        bj_position, bj_momentum = bj_step_jit(
            subkey, bj_position, bj_momentum, (X_all_jax, y_all_jax)
        )
        bj_all_params = bj_all_params.at[i + 1].set(bj_position)

    bj_sghmc_time = time() - bj_sghmc_time

    log_posts_bj = jnp.array(
        [
            log_posterior_normalized(samp, (X_all_jax, y_all_jax))
            for samp in bj_all_params[1:]
        ]
    )

    bj_all_params = bj_all_params[N_warmup_bj + 1 :: thinning_bj]

    samples = samples.at[r].set(bj_all_params)
    log_posts = log_posts.at[r].set(log_posts_bj)
    times = times.at[r].set(bj_sghmc_time)


save_dict = {
    "samples": np.array(samples),
    "log_posts": np.array(log_posts),
    "times": np.array(times),
}

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)
