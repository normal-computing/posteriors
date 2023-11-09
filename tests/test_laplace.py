import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector

from uqlib import laplace


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def normal_log_prior(p):
    return Normal(0, 1).log_prob(p).sum(dim=-1)


def normal_log_likelihood(y, y_pred):
    return Normal(y_pred, 1).log_prob(y).sum(dim=-1)


def test_fit_diagonal_hessian():
    model = TestModel()

    p = parameters_to_vector(model.parameters())

    xs = torch.randn(100, 10)
    ys = model(xs)

    dataloader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=20,
    )

    diag_hess = laplace.fit_diagonal_hessian(
        model, normal_log_prior, normal_log_likelihood, dataloader
    )

    assert diag_hess.shape == p.shape
