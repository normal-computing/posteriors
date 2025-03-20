from functools import partial
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call

from posteriors import tree_size, empirical_fisher, diag_normal_log_prob
from posteriors.laplace import dense_fisher

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


def test_dense_fisher_vmap():
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

    log_posterior_per_sample = torch.vmap(log_posterior, in_dims=(None, 0))

    params = dict(model.named_parameters())

    # Test inplace = False
    transform = dense_fisher.build(log_posterior)
    laplace_state = transform.init(params)
    laplace_state_prec_init = laplace_state.prec
    for batch in dataloader:
        laplace_state, _ = transform.update(laplace_state, batch, inplace=False)

    num_params = tree_size(params)
    expected = torch.zeros((num_params, num_params))
    for x, y in zip(xs, ys):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        with torch.no_grad():
            fisher = empirical_fisher(
                lambda p: log_posterior_per_sample(p, (x, y)),
                has_aux=True,
                normalize=False,
            )(params)[0]

        expected += fisher

    assert torch.allclose(expected, laplace_state.prec, atol=1e-5)
    assert not torch.allclose(laplace_state.prec, laplace_state_prec_init)

    # Also check full batch
    laplace_state_fb = transform.init(params)
    laplace_state_fb, _ = transform.update(laplace_state_fb, (xs, ys))

    assert torch.allclose(expected, laplace_state_fb.prec, atol=1e-5)

    #  Test per_sample
    log_posterior_per_sample = partial(log_posterior_n, model=model, n_data=len(xs))
    transform_ps = dense_fisher.build(log_posterior_per_sample, per_sample=True)
    laplace_state_ps = transform_ps.init(params)
    for batch in dataloader:
        laplace_state_ps, _ = transform_ps.update(
            laplace_state_ps,
            batch,
        )

    assert torch.allclose(laplace_state_ps.prec, laplace_state_fb.prec, atol=1e-5)


def test_dense_fisher_inplace():
    model = TestModel()

    xs = torch.randn(100, 10)
    ys = model(xs)

    batch = (xs, ys)

    def log_posterior(p, b):
        return log_posterior_n(p, b, model, len(xs))[0].mean(), torch.tensor([])

    params = dict(model.named_parameters())

    # Test inplace = False
    transform = dense_fisher.build(log_posterior)

    verify_inplace_update(transform, params, batch)


def test_dense_fisher_sample():
    torch.manual_seed(42)

    mean, cov = get_multivariate_normal_log_prob(dim=3)[1]

    state = dense_fisher.init(mean, cov.inverse())

    num_samples = 10000
    samples = dense_fisher.sample(state, (num_samples,))

    samples_mean = samples.mean(dim=0)
    samples_cov = torch.cov(samples.T)

    assert torch.allclose(samples_cov, cov, atol=1e-1)
    assert torch.allclose(samples_mean, state.params, atol=1e-1)
