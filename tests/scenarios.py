from typing import Any
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from optree import tree_leaves
from optree.integration.torch import tree_ravel

from posteriors.types import TensorTree, Transform, LogProbFn


class TestModel(nn.Module):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestLanguageModel(nn.Module):
    __test__ = False

    def __init__(self, vocab_size=1000, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.device = "cpu"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        hidden = self.linear(embedded)
        logits = self.output_layer(hidden)
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(logits.size())
        logits = logits * expanded_attention_mask
        return {"logits": logits}


def get_multivariate_normal_log_prob(
    dim: int,
) -> tuple[LogProbFn, tuple[torch.Tensor, torch.Tensor]]:
    mean = torch.randn(dim)
    sqrt_cov = torch.randn(dim, dim)
    cov = sqrt_cov @ sqrt_cov.T
    chol_cov = torch.linalg.cholesky(cov)  # Lower triangular with positive diagonal

    def log_prob(p, batch):
        p_flat = tree_ravel(p)[0]
        lp = MultivariateNormal(mean, scale_tril=chol_cov).log_prob(p_flat)
        return lp, torch.tensor([])

    return log_prob, (mean, cov)


def run_transform(
    transform: Transform,
    params: TensorTree,
    n_steps: int,
) -> TensorTree:
    state = transform.init(params)

    all_states = torch.unsqueeze(state, 0)

    for _ in range(n_steps):
        state, _ = transform.update(state, None)
        all_states = torch.cat((all_states, torch.unsqueeze(state, 0)), dim=0)

    return all_states


def cumulative_mean_and_cov(xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = xs.shape
    out_means = torch.zeros((n, d))
    out_covs = torch.zeros((n, d, d))

    out_means[0] = xs[0]
    out_covs[0] = torch.eye(d)

    for i in range(1, n):
        n = i + 1
        out_means[i] = out_means[i - 1] * n / (n + 1) + xs[i] / (n + 1)

        delta_n = xs[i] - out_means[i - 1]
        out_covs[i] = (
            out_covs[i - 1] * (n - 2) / (n - 1) + torch.outer(delta_n, delta_n) / n
        )

    return out_means, out_covs


def kl_gaussians(
    m0: torch.Tensor, c0: torch.Tensor, m1: torch.Tensor, c1: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (
        torch.trace(c1.inverse() @ c0)
        + torch.dot(m1 - m0, c1.inverse() @ (m1 - m0))
        - len(m0)
        + torch.log(torch.linalg.det(c1) / torch.linalg.det(c0))
    )


def verify_inplace_update(transform: Transform, params: TensorTree, batch: Any):
    init_state = transform.init(params)

    # One step not inplace
    torch.manual_seed(42)
    state_not_inplace, _ = transform.update(init_state, batch)

    # One step in place = same as not inplace
    torch.manual_seed(42)
    state_inplace, _ = transform.update(init_state, batch, inplace=True)

    # Iterate through leaves of params and check they are the same
    for init_leaf, not_inplace_leaf, inplace_leaf in zip(
        tree_leaves(init_state.to_dict()),
        tree_leaves(state_not_inplace.to_dict()),
        tree_leaves(state_inplace.to_dict()),
    ):
        assert torch.equal(inplace_leaf, not_inplace_leaf)
        assert torch.equal(init_leaf, not_inplace_leaf)
