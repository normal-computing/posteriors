import torch
import torch.nn as nn
from torch.distributions import Normal

from uqlib.vi import diagonal_nelbo
from uqlib.utils import tree_map


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def normal_log_prior(p: dict):
    output = 0
    for ptemp in p.values():
        output += Normal(0, 1).log_prob(ptemp).sum()
    return output


def normal_log_likelihood(y, y_pred):
    return Normal(y_pred, 1).log_prob(y).sum(dim=-1)


def test_nelbo():
    # Define a simple model
    model = TestModel()

    # Create a batch of data
    batch = (torch.randn(10, 10), torch.randn(10, 1))

    # Create mu and log_sigma_diag
    mu = tree_map(lambda x: torch.randn_like(x), model.state_dict())
    log_sigma_diag = tree_map(lambda x: torch.randn_like(x), mu)

    # Call the nelbo function
    loss = diagonal_nelbo(
        model, normal_log_prior, normal_log_likelihood, batch, mu, log_sigma_diag, 1000
    )

    loss2 = diagonal_nelbo(
        model, normal_log_prior, normal_log_likelihood, batch, mu, log_sigma_diag, 1000
    )

    # Check if the loss is a scalar and has the correct type
    assert loss.dim() == 0
    assert loss.dtype == torch.float32
    assert torch.isclose(loss, loss2, atol=1e0)
