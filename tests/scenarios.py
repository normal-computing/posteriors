from typing import Any, Tuple
import torch
import torch.nn as nn
from optree import tree_map

from uqlib.utils import diag_normal_log_prob, diag_normal_sample
from uqlib.types import TensorTree


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


def batch_normal_log_prob(
    p: dict, batch: Any, mean: dict, sd_diag: dict
) -> Tuple[torch.Tensor, TensorTree]:
    return diag_normal_log_prob(p, mean, sd_diag), torch.tensor([])


def gaussian_mixture_log_prob(
    p: dict, batch: Any, means: dict, diag_sds: dict, weights: dict
) -> Tuple[torch.Tensor, TensorTree]:
    log_mixture_probs = torch.vmap(diag_normal_log_prob, in_dims=(None, 0, 0))(
        p, means, diag_sds
    )
    return torch.logsumexp(log_mixture_probs + torch.log(weights)), torch.tensor([])


def gaussian_mixture_sample(
    means: TensorTree, diag_sds: TensorTree, weights: torch.Tensor, n_samples: int
) -> TensorTree:
    component_indices = torch.multinomial(weights, n_samples, replacement=True)

    def get_mean(i):
        return tree_map(lambda x: x[i], means)

    def get_sd(i):
        return tree_map(lambda x: x[i], diag_sds)

    return tree_map(
        lambda i: diag_normal_sample(get_mean(i), get_sd(i)), component_indices
    )
