import torch
import torch.nn as nn
from optree import tree_map

from uqlib import (
    hessian_diag,
    model_to_function,
    diag_normal_log_prob,
    diag_normal_sample,
)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestLanguageModel(nn.Module):
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


def test_model_to_function():
    model = TestModel()

    xs = torch.randn(100, 10)

    func_model = model_to_function(model)

    output = model(xs)
    func_output = func_model(dict(model.named_parameters()), xs)

    assert torch.allclose(output, func_output)

    lm = TestLanguageModel()

    input_ids = torch.randint(1000, (10, 10))
    attention_mask = torch.ones_like(input_ids)

    func_lm = model_to_function(lm)

    output = lm(input_ids, attention_mask)

    func_output1 = func_lm(
        dict(lm.named_parameters()), input_ids=input_ids, attention_mask=attention_mask
    )

    func_output2 = func_lm(dict(lm.named_parameters()), input_ids, attention_mask)

    assert type(output) == type(func_output1) == type(func_output2)
    assert torch.allclose(output["logits"], func_output1["logits"])
    assert torch.allclose(output["logits"], func_output2["logits"])


def test_hessian_diag():
    # Test with a constant function
    def const_fn(_):
        return torch.tensor(3.0)

    hessian_diag_func = hessian_diag(const_fn)
    x = torch.tensor([1.0, 2.0])
    result = hessian_diag_func(x)
    expected = torch.zeros_like(x)
    assert torch.equal(result, expected)

    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    result = hessian_diag_func(x)
    expected = tree_map(lambda v: torch.zeros_like(v), x)
    for key in result:
        assert torch.equal(result[key], expected[key])

    # Test with a linear function
    def linear_fn(x):
        return x["a"].sum() + x["b"].sum()

    hessian_diag_func = hessian_diag(linear_fn)
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    result = hessian_diag_func(x)
    for key in result:
        assert torch.equal(result[key], expected[key])

    # Test with a quadratic function
    def quad_fn(x):
        return (x["a"] ** 2).sum() + (x["b"] ** 2).sum()

    hessian_diag_func = hessian_diag(quad_fn)
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    result = hessian_diag_func(x)
    expected = tree_map(lambda v: 2 * torch.ones_like(v), x)
    for key in result:
        assert torch.equal(result[key], expected[key])


def test_diag_normal_log_prob():
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = {"a": torch.tensor([0.1, 0.2]), "b": torch.tensor([0.3, 0.4])}
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}

    mean_flat = torch.stack([mean["a"], mean["b"]]).flatten()
    sd_flat = torch.stack([sd_diag["a"], sd_diag["b"]]).flatten()
    x_flat = torch.stack([x["a"], x["b"]]).flatten()

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = torch.distributions.Normal(mean_flat, sd_flat).log_prob(x_flat).sum()

    assert torch.allclose(result, expected)


def test_diag_normal_sample():
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = {"a": torch.tensor([0.1, 0.2]), "b": torch.tensor([0.3, 0.4])}

    torch.manual_seed(42)
    result = diag_normal_sample(mean, sd_diag)
    torch.manual_seed(42)
    expected = tree_map(
        lambda m, sd: torch.distributions.Normal(m, sd).sample(), mean, sd_diag
    )

    for key in result:
        assert torch.equal(result[key], expected[key])

    n_samps = 1000
    result = diag_normal_sample(mean, sd_diag, sample_shape=(n_samps,))
    result_mean = tree_map(lambda v: v.mean(dim=0), result)
    result_std = tree_map(lambda v: v.std(dim=0), result)

    for key in result_mean:
        assert torch.allclose(result_mean[key], mean[key], atol=1e-1)
        assert torch.allclose(result_std[key], sd_diag[key], atol=1e-1)
