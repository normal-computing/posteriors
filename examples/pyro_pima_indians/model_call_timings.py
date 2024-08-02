from functools import partial
from time import time
import jax
import torch

from examples.pyro_pima_indians.model import load_data, load_model, load_jax_model

comp = True


X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

model, log_posterior_torch = load_model(num_data)
log_posterior_jax = load_jax_model(num_data)


initial_params_torch = torch.zeros(dim)
log_posterior_torch_c = (
    torch.compile(
        lambda p: log_posterior_torch(p, (X_all, y_all))[0], mode="reduce-overhead"
    )
    if comp
    else lambda p: log_posterior_torch(p, (X_all, y_all))[0]
)
grad_log_posterior_torch = (
    torch.compile(torch.func.grad(log_posterior_torch_c), mode="reduce-overhead")
    if comp
    else torch.func.grad(log_posterior_torch_c)
)


X_all_jax = jax.numpy.array(X_all.numpy())
y_all_jax = jax.numpy.array(y_all.numpy())

initial_params_jax = jax.numpy.zeros(dim)
log_posterior_jax_jit = (
    jax.jit(partial(log_posterior_jax, batch=(X_all_jax, y_all_jax)))
    if comp
    else partial(log_posterior_jax, batch=(X_all_jax, y_all_jax))
)
grad_log_posterior_jax = (
    jax.jit(jax.grad(log_posterior_jax_jit))
    if comp
    else jax.grad(log_posterior_jax_jit)
)

# Compile
_ = log_posterior_torch_c(initial_params_torch)
_ = log_posterior_jax_jit(initial_params_jax)
_ = grad_log_posterior_torch(initial_params_torch)
_ = grad_log_posterior_jax(initial_params_jax)


def time_func(f, v, rep=100):
    start = time()
    for i in range(rep):
        f(v + i)
    end = time()
    return (end - start) / rep * 1000


def time_func_jax(f, v, rep=100):
    start = time()
    for i in range(rep):
        f(v + i).block_until_ready()
    end = time()
    return (end - start) / rep * 1000


with torch.no_grad():
    lp_torch_time = time_func(log_posterior_torch_c, initial_params_torch)
    glp_torch_time = time_func(grad_log_posterior_torch, initial_params_torch)

lp_jax_time = time_func_jax(log_posterior_jax_jit, initial_params_jax)
glp_jax_time = time_func_jax(grad_log_posterior_jax, initial_params_jax)


print(f"Log posterior torch: {lp_torch_time}")
print(f"Grad log posterior torch: {glp_torch_time}")
print(f"Log posterior jax: {lp_jax_time}")
print(f"Grad log posterior jax: {glp_jax_time}")

print("Torch/Jax LP: ", lp_torch_time / lp_jax_time)
print("Torch/Jax GLP: ", glp_torch_time / glp_jax_time)
