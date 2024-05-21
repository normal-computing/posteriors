from functools import partial
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from optree import tree_map
from optree.integration.torch import tree_ravel

from posteriors.laplace import dense_ggn

from tests.scenarios import TestModel


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
        laplace_state = transform.update(laplace_state, batch, inplace=False)

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
    laplace_state_fb = transform.update(laplace_state_fb, (xs, ys))

    assert torch.allclose(expected, laplace_state_fb.prec, atol=1e-5)

    # Test inplace = True
    laplace_state = transform.init(params)
    laplace_state_prec_init = laplace_state.prec
    for batch in dataloader:
        laplace_state = transform.update(laplace_state, batch, inplace=True)

    assert torch.allclose(expected, laplace_state.prec, atol=1e-5)
    assert torch.allclose(laplace_state.prec, laplace_state_prec_init)

    # Test sampling
    num_samples = 10000
    laplace_state.prec = laplace_state.prec + 0.1 * torch.eye(
        flat_params.shape[0]
    )  # regularize to ensure PSD and reduce variance

    mean_copy = tree_map(lambda x: x.clone(), laplace_state.params)
    sd_flat = torch.diag(torch.linalg.inv(laplace_state.prec)).sqrt()

    samples = dense_ggn.sample(laplace_state, (num_samples,))

    samples_mean = tree_map(lambda x: x.mean(dim=0), samples)
    samples_sd = tree_map(lambda x: x.std(dim=0), samples)
    samples_sd_flat = tree_ravel(samples_sd)[0]

    for key in samples_mean:
        assert samples[key].shape[0] == num_samples
        assert samples[key].shape[1:] == samples_mean[key].shape
        assert torch.allclose(samples_mean[key], laplace_state.params[key], atol=1e-1)
        assert torch.allclose(mean_copy[key], laplace_state.params[key])

    assert torch.allclose(sd_flat, samples_sd_flat, atol=1e-1)
