import pickle
import pprint
import torch
import numpy as np
from tqdm import tqdm

from examples.pyro_pima_indians.model import load_data, load_model
from examples.pyro_pima_indians.ksd import ksd

name_dict = {
    "Pyro (NUTS)": "examples/pyro_pima_indians/results/pyro.pkl",
    "BlackJAX (SGHMC, Full Batch)": "examples/pyro_pima_indians/results/blackjax_sghmc.pkl",
    "posteriors (SGHMC, Full Batch)": "examples/pyro_pima_indians/results/posteriors_sghmc_None.pkl",
    "posteriors (SGHMC, Batch Size=32)": "examples/pyro_pima_indians/results/posteriors_sghmc_32.pkl",
    "posteriors (VI, Full Batch)": "examples/pyro_pima_indians/results/posteriors_vi_None.pkl",
    "posteriors (VI, Batch Size=32)": "examples/pyro_pima_indians/results/posteriors_vi_32.pkl",
    "posteriors (Parallel SGHMC, Batch Size=32)": "examples/pyro_pima_indians/results/posteriors_sghmc_parallel_32.pkl",
}


sample_dict = {}
time_dict = {}
for key, dir in name_dict.items():
    with open(dir, "rb") as f:
        save_dict = pickle.load(f)
        sample_dict[key] = save_dict["samples"]
        time_dict[key] = save_dict["times"]


time_summs_dict = {
    key: (time_dict[key].mean(), time_dict[key].std()) for key in time_dict
}

# Print the mean and standard deviation of the time taken for each method
pprint.pprint(time_summs_dict)


# Stein discrepancy
# The Stein discrepancy is a measure of the difference between a collection of samples and a log posterior function.

ksd_batchsize = 100
ksd_save_dir = "examples/pyro_pima_indians/results/ksd.pickle"

X_all, y_all = load_data()
dim = X_all.shape[1]
num_data = X_all.shape[0]

model, log_posterior_normalized = load_model(num_data)


def grad_log_posterior_normalized(params):
    return torch.func.grad(lambda p: log_posterior_normalized(p, (X_all, y_all))[0])(
        params
    )


def ksd_via_grads(samples, batchsize=None):
    gradients = torch.stack([grad_log_posterior_normalized(s) for s in samples])
    return ksd(samples, gradients, batchsize=batchsize)


ksd_dict = {}
for key, samples in tqdm(sample_dict.items()):
    ksd_dict[key] = np.array(
        [ksd_via_grads(torch.tensor(s), batchsize=ksd_batchsize) for s in samples]
    )

ksd_summs_dict = {key: (ksd_dict[key].mean(), ksd_dict[key].std()) for key in ksd_dict}


# Print the mean and standard deviation of the KSD for each method
pprint.pprint(ksd_summs_dict)

with open(ksd_save_dir, "wb") as f:
    pickle.dump(ksd_summs_dict, f)
