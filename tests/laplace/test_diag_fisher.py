from functools import partial
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from optree.integration.torch import tree_ravel
from optree import tree_map

from posteriors.laplace import diag_fisher
from posteriors import diag_normal_log_prob

from tests.scenarios import TestModel, get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update


def normal_log_likelihood(y, y_pred):
    return (
        Normal(y_pred, 1, validate_args=False).log_prob(y).sum(dim=-1)
    )  # validate args introduces control flows not yet supported in torch.func.vmap


def log_posterior_n(params, batch, model, n_data):
    y_pred = functional_call(model, params, batch[0])
    return diag_normal_log_prob(params, mean=0.0, sd_diag=1.0) + normal_log_likelihood(
        batch[1], y_pred
    ) * n_data, torch.tensor([])


def test_diag_fisher_vmap():
    torch.manual_seed(42)
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    def log_posterior(p, b):
        return log_posterior_n(p, b, model, len(xs))[0].mean(), torch.tensor([])

    params = dict(model.named_parameters())

    # Test inplace = False
    transform = diag_fisher.build(log_posterior)
    laplace_state = transform.init(params)
    laplace_state_prec_diag_init = tree_map(lambda x: x, laplace_state.prec_diag)
    for batch in dataloader:
        laplace_state, _ = transform.update(laplace_state, batch, inplace=False)

    expected = tree_map(lambda x: torch.zeros_like(x), params)
    for x, y in zip(xs, ys):
        with torch.no_grad():
            g, _ = torch.func.grad(lambda p: log_posterior(p, (x, y)), has_aux=True)(
                params
            )
        expected = tree_map(lambda x, y: x + y**2, expected, g)

    for key in expected:
        assert torch.allclose(expected[key], laplace_state.prec_diag[key], atol=1e-5)
        assert not torch.allclose(
            laplace_state.prec_diag[key], laplace_state_prec_diag_init[key]
        )

    # Also check full batch
    laplace_state_fb = transform.init(params)
    laplace_state_fb, _ = transform.update(laplace_state_fb, (xs, ys))

    for key in expected:
        assert torch.allclose(expected[key], laplace_state_fb.prec_diag[key], atol=1e-5)

    #  Test per_sample
    log_posterior_per_sample = partial(log_posterior_n, model=model, n_data=len(xs))
    transform_ps = diag_fisher.build(log_posterior_per_sample, per_sample=True)
    laplace_state_ps = transform_ps.init(params)
    for batch in dataloader:
        laplace_state_ps, _ = transform_ps.update(
            laplace_state_ps,
            batch,
            inplace=False,
        )

    for key in expected:
        assert torch.allclose(
            laplace_state_ps.prec_diag[key], laplace_state_fb.prec_diag[key], atol=1e-5
        )


def test_diag_fisher_inplace():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    batch = (xs, ys)

    def log_posterior(p, b):
        return log_posterior_n(p, b, model, len(xs))[0].mean(), torch.tensor([])

    params = dict(model.named_parameters())

    verify_inplace_update(diag_fisher.build(log_posterior), params, batch)


def test_diag_fisher_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]
    cov_diag = torch.diag(cov)

    state = diag_fisher.init(mean, 1 / cov_diag)

    num_samples = 10000
    samples = diag_fisher.sample(state, (num_samples,))

    flat_samples = torch.vmap(lambda s: tree_ravel(s)[0])(samples)
    samples_cov = torch.cov(flat_samples.T)
    samples_sd = torch.sqrt(torch.diag(samples_cov))

    mean_copy = state.params.clone()
    samples_mean = flat_samples.mean(dim=0)

    assert torch.allclose(samples_sd, torch.sqrt(1 / state.prec_diag), atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
    assert not torch.allclose(samples_mean, mean_copy)
