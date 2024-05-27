import torch
from optree import tree_map
from torch.distributions import MultivariateNormal
from optree.integration.torch import tree_ravel
from posteriors.tree_utils import tree_size
from posteriors import ekf


def test_ekf_dense():
    torch.manual_seed(42)
    target_mean = {"a": torch.randn(2, 1), "b": torch.randn(1, 1)}
    num_params = tree_size(target_mean)
    A = torch.randn(num_params, num_params)
    target_cov = torch.mm(A.t(), A)

    dist = MultivariateNormal(tree_ravel(target_mean)[0], covariance_matrix=target_cov)

    def log_prob(p, b):
        return dist.log_prob(tree_ravel(p)[0]).sum(), torch.Tensor([])

    init_mean = tree_map(lambda x: torch.zeros_like(x, requires_grad=True), target_mean)
    batch = torch.arange(3).reshape(-1, 1)
    n_steps = 1000
    transform = ekf.dense_fisher.build(log_prob, lr=1e-1)

    # Test inplace = False
    state = transform.init(init_mean)
    log_liks = []
    for _ in range(n_steps):
        state = transform.update(state, batch, inplace=False)
        log_liks.append(state.log_likelihood.item())

    assert log_liks[0] < log_liks[-1]

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=1e-1)
        assert not torch.allclose(state.params[key], init_mean[key])

    # Test inplace = True
    state = transform.init(init_mean)
    log_liks = []
    for _ in range(n_steps):
        state = transform.update(state, batch, inplace=True)
        log_liks.append(state.log_likelihood.item())

    for key in state.params:
        assert torch.allclose(state.params[key], target_mean[key], atol=1e-1)
        assert not torch.allclose(state.params[key], init_mean[key])

    # Test sample
    num_samples = 1000
    samples = ekf.dense_fisher.sample(state, (num_samples,))

    # Reshaping the samples after raveling them does not return the correct order. This is because,
    # on raveling, all the samples under "a" are added to the list left to right, only then followed by
    # the samples under parameter "b". Reshaping therefore has whole rows of "a" samples before "b" samples begin.
    # Really, we want each 2D "a" sample followed by a 1D "b" sample, each 3-column row containing a sample of each param in this way.
    # Therefore, we have to carefully column stack the result of raveling each parameter and reshaping them.
    # This is all to aid in computing the sample covariance.
    k = list(samples.keys())[0]  # get first sample to start column stacking
    samples_copy = tree_ravel(samples[k])[0].reshape(num_samples, samples[k].shape[1])
    for k in list(samples.keys())[1:]:
        v = tree_ravel(samples[k])[0].reshape(num_samples, samples[k].shape[1])
        samples_copy = torch.column_stack((samples_copy, v))
    samples_cov = torch.cov(samples_copy.T)

    mean_copy = tree_map(lambda x: x.clone(), state.params)
    samples_mean = tree_map(lambda x: x.mean(dim=0), samples)

    assert torch.allclose(samples_cov, state.cov, atol=1e-1)
    for key in samples_mean:
        assert torch.allclose(samples_mean[key], state.params[key], atol=1e-1)
        assert not torch.allclose(samples_mean[key], mean_copy[key])
