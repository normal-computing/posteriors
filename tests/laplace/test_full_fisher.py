from functools import partial
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from uqlib.utils import tree_size
from optree.integration.torch import ravel_pytree

from uqlib.laplace import full_fisher


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


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

    xs = torch.randn(10, 4)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=4,
    )

    def log_posterior(p, b):
        return log_posterior_n(p, b, model, len(xs))[0].mean(), torch.tensor([])

    params = dict(model.named_parameters())

    transform = full_fisher.build(log_posterior)
    laplace_state = transform.init(params)
    for batch in dataloader:
        laplace_state = transform.update(laplace_state, batch)
        # print(laplace_state.prec)

    print("finished full fisher")
    num_params = tree_size(params)
    expected = torch.zeros((num_params, num_params))
    for x, y in zip(xs, ys):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        with torch.no_grad():
            jac, _ = torch.func.jacrev(log_posterior, has_aux=True)(params, (x, y))
            flat_jac, _ = ravel_pytree(jac)
            flat_jac = flat_jac.reshape(x.shape[0], -1)
            fisher = flat_jac.T @ flat_jac
        expected += fisher

    print(expected)
    print(laplace_state.prec)
    assert torch.allclose(expected, laplace_state.prec, atol=1e-4)

    # Also check full batch
    laplace_state_fb = transform.init(params)
    # print(laplace_state_fb.prec)
    laplace_state_fb = transform.update(laplace_state_fb, (xs, ys))
    print(laplace_state_fb.prec)

    assert torch.allclose(expected, laplace_state_fb.prec, atol=1e-4)

    #  Test per_sample
    log_posterior_per_sample = partial(log_posterior_n, model=model, n_data=len(xs))
    transform_ps = full_fisher.build(log_posterior_per_sample, per_sample=True)
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

    # Test not inplace
    laplace_state_ip_false = transform.update(
        laplace_state_ip,
        batch,
        inplace=False,
    )

    assert not torch.allclose(
        laplace_state_ip_false.prec,
        laplace_state_ip.prec,
        atol=1e-8,
    )


test_full_fisher_vmap()
