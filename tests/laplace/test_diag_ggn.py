from functools import partial
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from optree import tree_map
from optree.integration.torch import tree_ravel

from posteriors.laplace import diag_ggn

from tests.scenarios import TestModel, get_multivariate_normal_log_prob
from tests.utils import verify_inplace_update


def normal_log_likelihood(y_pred, batch):
    y = batch[1]
    return (
        Normal(y_pred, 1, validate_args=False).log_prob(y).sum()
    )  # validate args introduces control flows not yet supported in torch.func.vmap


def forward_m(params, b, model):
    y_pred = functional_call(model, params, b[0])
    return y_pred, torch.tensor([])


def test_diag_ggn_vmap():
    torch.manual_seed(42)
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    forward = partial(forward_m, model=model)

    params = dict(model.named_parameters())

    # Test inplace = False
    transform = diag_ggn.build(forward, normal_log_likelihood)
    laplace_state = transform.init(params)
    laplace_state_prec_diag_init = tree_map(lambda x: x, laplace_state.prec_diag)
    for batch in dataloader:
        laplace_state, _ = transform.update(laplace_state, batch, inplace=False)

    flat_params, unravel_fn = tree_ravel(params)

    expected = tree_map(lambda x: torch.zeros_like(x), params)
    for x, y in zip(xs, ys):
        with torch.no_grad():
            z = forward(params, (x, y))[0]
            J = torch.func.jacrev(lambda fp: forward(unravel_fn(fp), (x, y)))(
                flat_params
            )[0]
            H = torch.func.hessian(lambda zt: normal_log_likelihood(zt, (x, y)))(z)
            G = J.T @ H @ J
        expected = tree_map(lambda x, y: x - y, expected, unravel_fn(torch.diag(G)))

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


def test_diag_ggn_inplace():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    batch = (xs, ys)
    forward = partial(forward_m, model=model)
    transform = diag_ggn.build(forward, normal_log_likelihood)

    params = dict(model.named_parameters())
    verify_inplace_update(transform, params, batch)


def test_diag_ggn_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]
    cov_diag = torch.diag(cov)

    state = diag_ggn.init(mean, 1 / cov_diag)

    num_samples = 10000
    samples = diag_ggn.sample(state, (num_samples,))

    flat_samples = torch.vmap(lambda s: tree_ravel(s)[0])(samples)
    samples_cov = torch.cov(flat_samples.T)
    samples_sd = torch.sqrt(torch.diag(samples_cov))

    mean_copy = state.params.clone()
    samples_mean = flat_samples.mean(dim=0)

    assert torch.allclose(samples_sd, torch.sqrt(1 / state.prec_diag), atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
    assert not torch.allclose(samples_mean, mean_copy)
