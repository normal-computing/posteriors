import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from optree.integrations.torch import tree_ravel
from posteriors.types import LogProbFn


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
    cov = sqrt_cov @ sqrt_cov.T * 0.3 + torch.eye(dim) * 0.7
    chol_cov = torch.linalg.cholesky(cov)  # Lower triangular with positive diagonal

    def log_prob(p, batch):
        p_flat = tree_ravel(p)[0]
        lp = MultivariateNormal(
            mean, scale_tril=chol_cov, validate_args=False
        ).log_prob(p_flat)
        return lp, torch.tensor([])

    return log_prob, (mean, cov)
