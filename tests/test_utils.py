import torch
import torch.nn as nn
from optree import tree_map, tree_flatten

from uqlib import (
    model_to_function,
    linearized_forward_diag,
    hessian_diag,
    diag_normal_log_prob,
    diag_normal_sample,
    extract_requires_grad,
    insert_requires_grad,
    insert_requires_grad_,
    extract_requires_grad_and_func,
    inplacify,
    per_samplify,
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


def test_linearized_forward_diag():
    vocab_size = 50
    batch_size = 3
    lm = TestLanguageModel(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)

    lm_functional = model_to_function(lm)

    input_ids = torch.randint(vocab_size, (batch_size, batch_size))
    attention_mask = torch.ones_like(input_ids)
    params = dict(lm.named_parameters())

    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward_func(p, batch):
        output = lm_functional(p, **batch)
        next_token_logits = output["logits"][:, -1, :]
        return next_token_logits, output

    sd_diag = tree_map(lambda x: torch.randn_like(x).abs(), params)

    mean, lin_cov_chol, output = linearized_forward_diag(
        forward_func, params, batch, sd_diag
    )

    assert mean.shape == (batch_size, vocab_size)
    assert lin_cov_chol.shape == (batch_size, vocab_size, vocab_size)
    assert torch.allclose(output["logits"][:, -1, :], mean)

    lin_cov = lin_cov_chol @ lin_cov_chol.transpose(-1, -2)

    jac = torch.func.jacrev(forward_func, has_aux=True)(params, batch)[0]
    jac_flat = tree_flatten(jac)[0]
    jac_flat = torch.cat([x.flatten(start_dim=2) for x in jac_flat], dim=2)

    sd_diag_flat = tree_flatten(sd_diag)[0]
    sd_diag_flat = torch.cat([x.flatten() for x in sd_diag_flat])

    for i in range(batch_size):
        jac_i = jac_flat[i]
        delta = jac_i @ torch.diag(sd_diag_flat**2) @ jac_i.T
        assert torch.allclose(lin_cov[i], delta, atol=1e-5)
        assert torch.allclose(lin_cov[i], lin_cov_chol[i] @ lin_cov_chol[i].T)


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


def test_extract_requires_grad():
    params = {
        "a": torch.tensor([1.0, 2.0], requires_grad=True),
        "b": torch.tensor([3.0, 4.0], requires_grad=False),
    }

    result = extract_requires_grad(params)

    expected = {
        "a": torch.tensor([1.0, 2.0], requires_grad=True),
        "b": torch.tensor([], requires_grad=False),
    }

    for key in expected:
        assert torch.equal(result[key], expected[key])
        assert result[key].requires_grad == expected[key].requires_grad


def test_insert_requires_grad():
    params = {
        "a": torch.tensor([1.0, 2.0], requires_grad=True),
        "b": torch.tensor([3.0, 4.0], requires_grad=False),
    }

    params_copy = {k: v.clone() for k, v in params.items()}

    sub_params = {
        "a": torch.tensor([5.0, 6.0], requires_grad=True),
        "b": torch.tensor([]),
    }

    params2 = insert_requires_grad(params, sub_params)

    expected = {
        "a": torch.tensor([5.0, 6.0], requires_grad=True),
        "b": torch.tensor([3.0, 4.0], requires_grad=False),
    }

    for key in expected:
        assert torch.equal(params2[key], expected[key])
        assert params2[key].requires_grad == expected[key].requires_grad
        assert torch.equal(params[key], params_copy[key])

    params = insert_requires_grad_(params, sub_params)

    for key in expected:
        assert torch.equal(params[key], expected[key])
        assert params[key].requires_grad == expected[key].requires_grad


def test_extract_requires_grad_func():
    params = {
        "a": torch.tensor([1.0, 2.0], requires_grad=True),
        "b": torch.tensor([3.0, 4.0], requires_grad=False),
    }

    def func(params):
        return params["a"].sum() + params["b"].sum()

    sub_params, sub_func = extract_requires_grad_and_func(params, func, inplace=False)

    result = sub_func(sub_params)

    assert torch.equal(result, func(params))

    sub_params, sub_func = extract_requires_grad_and_func(params, func, inplace=True)

    sub_params["a"].data = torch.tensor([5.0, 6.0])

    result2 = sub_func(sub_params)

    params2 = insert_requires_grad(params, sub_params)

    assert torch.equal(result2, func(params2))
    assert torch.equal(params2["a"], sub_params["a"])
    assert torch.equal(params2["a"], params["a"])


def test_inplacify():
    def func(x):
        return x + 1

    x = torch.tensor(1.0)
    y = inplacify(func)(x)

    assert y == 2.0
    assert x == 2.0


def test_per_samplify():
    def func(p, b):
        return p + b, p

    p = torch.tensor(1.0)
    b = torch.tensor([1.0, 2.0])

    func1 = per_samplify(func)
    ra, rb = func1(p, b)
    expected_a = torch.tensor([[2.0], [3.0]])
    expected_b = torch.tensor([1.0, 1.0])
    assert torch.allclose(ra, expected_a)
    assert torch.allclose(rb, expected_b)
