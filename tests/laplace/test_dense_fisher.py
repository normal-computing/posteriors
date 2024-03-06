from functools import partial
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from uqlib.utils import tree_size, empirical_fisher
from optree.integration.torch import tree_ravel

from uqlib.laplace import dense_fisher
from tests.scenarios import TestModel


# Note this won't work if you replace torch.stack with torch.tensor
def normal_log_prior(p: dict):
    return torch.sum(
        torch.stack([Normal(0, 1).log_prob(ptemp).sum() for ptemp in p.values()])
    )


def normal_log_likelihood(y, y_pred):
    return (
        Normal(y_pred, 1, validate_args=False).log_prob(y).sum(dim=-1)
    )  # validate args introduces control flows not yet supported in torch.func.vmap


# def normal_log_likelihood(y, y_pred):
#     return (y - y_pred).square().sum(dim=-1)


def log_posterior_n(params, batch, model, n_data):
    y_pred = functional_call(model, params, batch[0])
    return normal_log_prior(params) + normal_log_likelihood(
        batch[1], y_pred
    ) * n_data, torch.tensor([])


def test_full_fisher_vmap():
    torch.manual_seed(42)
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=2,
    )

    def log_posterior(p, b):
        return log_posterior_n(p, b, model, len(xs))[0].mean(), torch.tensor([])

    params = dict(model.named_parameters())

    # Test inplace = False
    transform = dense_fisher.build(log_posterior)
    laplace_state = transform.init(params)
    laplace_state_prec_init = laplace_state.prec
    for batch in dataloader:
        laplace_state = transform.update(laplace_state, batch, inplace=False)

    num_params = tree_size(params)
    expected = torch.zeros((num_params, num_params))
    for x, y in zip(xs, ys):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        with torch.no_grad():
            fisher = empirical_fisher(log_posterior, params, (x, y))[0]

        expected += fisher

    assert torch.allclose(expected, laplace_state.prec, atol=1e-5)
    assert not torch.allclose(laplace_state.prec, laplace_state_prec_init)

    # Also check full batch
    laplace_state_fb = transform.init(params)
    laplace_state_fb = transform.update(laplace_state_fb, (xs, ys))

    assert torch.allclose(expected, laplace_state_fb.prec, atol=1e-5)

    #  Test per_sample
    log_posterior_per_sample = partial(log_posterior_n, model=model, n_data=len(xs))
    transform_ps = dense_fisher.build(log_posterior_per_sample, per_sample=True)
    laplace_state_ps = transform_ps.init(params)
    for batch in dataloader:
        laplace_state_ps = transform_ps.update(
            laplace_state_ps,
            batch,
        )

    assert torch.allclose(laplace_state_ps.prec, laplace_state_fb.prec, atol=1e-5)

    # Test inplace
    laplace_state_ip = transform.init(params)
    laplace_state_ip2 = transform.update(
        laplace_state_ip,
        batch,
        inplace=True,
    )

    assert torch.allclose(laplace_state_ip2.prec, laplace_state_ip.prec, atol=1e-8)

    # Test inplace = True
    transform = dense_fisher.build(log_posterior)
    laplace_state = transform.init(params)
    laplace_state_prec_diag_init = laplace_state.prec
    for batch in dataloader:
        laplace_state = transform.update(laplace_state, batch, inplace=True)

    assert torch.allclose(expected, laplace_state.prec, atol=1e-5)
    assert torch.allclose(laplace_state.prec, laplace_state_prec_diag_init, atol=1e-5)

    # Test sampling
    num_samples = 100000
    laplace_state.prec = laplace_state.prec + 0.1 * torch.eye(
        num_params
    )  # regularize to ensure PSD and reduce variance
    samples, _ = dense_fisher.sample(laplace_state, (num_samples,))

    expected_samples = torch.distributions.MultivariateNormal(
        loc=tree_ravel(laplace_state.mean)[0],
        precision_matrix=laplace_state.prec,
        validate_args=False,
    ).sample((num_samples,))

    assert torch.allclose(
        torch.mean(samples, dim=0), torch.mean(expected_samples, dim=0), atol=1e-1
    )
