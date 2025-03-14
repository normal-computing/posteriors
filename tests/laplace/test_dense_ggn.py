from functools import partial
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from optree.integration.torch import tree_ravel

from posteriors.laplace import dense_ggn
from tests.utils import verify_inplace_update
from tests.scenarios import TestModel, get_multivariate_normal_log_prob


def normal_log_likelihood(y_pred, batch):
    y = batch[1]
    return (
        Normal(y_pred, 1, validate_args=False).log_prob(y).sum()
    )  # validate args introduces control flows not yet supported in torch.func.vmap


def forward_m(params, b, model):
    y_pred = functional_call(model, params, b[0])
    return y_pred, torch.tensor([])


def test_ggn_vmap():
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
    transform = dense_ggn.build(forward, normal_log_likelihood)
    laplace_state = transform.init(params)
    laplace_state_prec_init = laplace_state.prec
    for batch in dataloader:
        laplace_state, _ = transform.update(laplace_state, batch, inplace=False)

    flat_params, unravel_fn = tree_ravel(params)

    expected = torch.zeros((flat_params.shape[0], flat_params.shape[0]))
    for x, y in zip(xs, ys):
        with torch.no_grad():
            z = forward(params, (x, y))[0]
            J = torch.func.jacrev(lambda fp: forward(unravel_fn(fp), (x, y)))(
                flat_params
            )[0]
            H = torch.func.hessian(lambda zt: normal_log_likelihood(zt, (x, y)))(z)
            G = J.T @ H @ J
        expected -= G

    assert torch.allclose(expected, laplace_state.prec, atol=1e-5)
    assert not torch.allclose(laplace_state.prec, laplace_state_prec_init)

    # Also check full batch
    laplace_state_fb = transform.init(params)
    laplace_state_fb, _ = transform.update(laplace_state_fb, (xs, ys))

    assert torch.allclose(expected, laplace_state_fb.prec, atol=1e-5)


def test_dense_ggn_inplace():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    batch = (xs, ys)
    forward = partial(forward_m, model=model)
    transform = dense_ggn.build(forward, normal_log_likelihood)

    params = dict(model.named_parameters())
    verify_inplace_update(transform, params, batch)


def test_dense_ggn_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]

    state = dense_ggn.init(mean, cov.inverse())

    num_samples = 10000
    samples = dense_ggn.sample(state, (num_samples,))

    samples_mean = samples.mean(dim=0)
    samples_cov = torch.cov(samples.T)

    assert torch.allclose(samples_cov, cov, atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
