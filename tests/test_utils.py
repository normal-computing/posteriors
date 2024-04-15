import pytest
import torch
from optree import tree_map, tree_flatten, tree_reduce
from optree.integration.torch import tree_ravel

from posteriors import (
    CatchAuxError,
    model_to_function,
    linearized_forward_diag,
    hvp,
    fvp,
    empirical_fisher,
    diag_normal_log_prob,
    diag_normal_sample,
    tree_size,
    tree_extract,
    tree_insert,
    tree_insert_,
    extract_requires_grad,
    insert_requires_grad,
    insert_requires_grad_,
    extract_requires_grad_and_func,
    inplacify,
    tree_map_inplacify_,
    flexi_tree_map,
    per_samplify,
    is_scalar,
)
from posteriors.utils import AUX_ERROR_MSG
from tests.scenarios import TestModel, TestLanguageModel


def test_CatchAuxError():
    def func(x):
        return x**2

    def func_aux(x):
        return x**2, None

    # Check AUX_ERROR_MSG is correct
    try:
        torch.func.grad(func, has_aux=True)(torch.tensor(1.0))
    except Exception as e:
        assert AUX_ERROR_MSG in str(e)

    with pytest.raises(RuntimeError) as e:
        with CatchAuxError():
            torch.func.grad(func, has_aux=True)(torch.tensor(1.0))

        assert "Auxiliary output not found" in str(e)

    with pytest.raises(RuntimeError) as e:
        with torch.no_grad(), CatchAuxError():
            torch.func.grad(func, has_aux=True)(torch.tensor(1.0))

        assert "Auxiliary output not found" in str(e)

    with CatchAuxError():
        torch.func.grad(func_aux, has_aux=True)(torch.tensor(1.0))

    with torch.no_grad(), CatchAuxError():
        torch.func.grad(func_aux, has_aux=True)(torch.tensor(1.0))


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


def test_hvp():
    def func(x):
        return (x**5).sum()

    x = torch.arange(1.0, 6.0)
    v = torch.ones_like(x)
    hvp_result = hvp(func, (x,), (v,))
    expected = torch.func.hessian(func)(x) @ v
    assert torch.allclose(hvp_result[0], torch.func.grad(func)(x))
    assert torch.allclose(hvp_result[1], expected)

    def func_and_aux(x):
        return (x**5).sum(), x**2

    hvp_result_aux = hvp(func_and_aux, (x,), (v,), has_aux=True)

    assert torch.allclose(hvp_result_aux[0], torch.func.grad(func)(x))
    assert torch.allclose(hvp_result_aux[1], expected)
    assert torch.allclose(hvp_result_aux[2], x**2)


def test_fvp():
    def func(x):
        return torch.stack([(x**5).sum(), (x**3).sum()])

    x = torch.arange(1.0, 6.0)
    v = torch.ones_like(x)

    # Test not normalized
    output, fvp_result = fvp(func, (x,), (v,), normalize=False)

    jac = torch.func.jacrev(func)(x)
    fisher = jac.T @ jac
    expected = fisher @ v
    assert torch.allclose(fvp_result, expected)
    assert torch.allclose(output, func(x))

    # Test normalize
    output_norm, fvp_result_norm = fvp(func, (x,), (v,))
    assert torch.allclose(fvp_result_norm, expected / 2)
    assert torch.allclose(output_norm, func(x))

    # Test aux, not normalized
    def func_aux(x):
        return torch.stack([(x**5).sum(), (x**3).sum()]), x

    output_aux, fvp_aux_result, aux = fvp(
        func_aux, (x,), (v,), has_aux=True, normalize=False
    )
    assert torch.allclose(fvp_aux_result, expected)
    assert torch.allclose(output_aux, func(x))
    assert torch.allclose(aux, x)

    # Test aux, normalized
    output_aux_norm, fvp_aux_result_norm, aux_norm = fvp(
        func_aux, (x,), (v,), has_aux=True, normalize=True
    )
    assert torch.allclose(fvp_aux_result_norm, expected / 2)
    assert torch.allclose(output_aux_norm, func(x))
    assert torch.allclose(aux_norm, x)


def test_empirical_fisher():
    # Test no aux
    def f(params, batch):
        return torch.squeeze(batch[1] - batch[0] @ params["weights"] - params["bias"])

    f_per_sample = torch.vmap(f, in_dims=(None, 0))

    num_samples = 2
    num_features = 4

    x = torch.randn(num_samples, num_features)
    y = torch.randn(
        num_samples,
    )
    params = {
        "weights": torch.randn(num_features, 1, requires_grad=True),
        "bias": torch.randn(1, requires_grad=True),
    }

    # Test not normalized
    batch = (x, y)
    fisher = empirical_fisher(lambda p: f_per_sample(p, batch), normalize=False)(params)
    expected_fisher = torch.zeros((num_features + 1, num_features + 1))
    for xs, ys in zip(x, y):
        g = torch.func.grad(f)(params, (xs, ys))
        g = tree_ravel(g)[0]
        expected_fisher += torch.outer(g, g)

    assert torch.allclose(fisher, expected_fisher, rtol=1e-5)

    # Test normalized
    fisher_norm = empirical_fisher(lambda p: f_per_sample(p, batch))(params)
    assert torch.allclose(fisher_norm, expected_fisher / num_samples, rtol=1e-5)

    # Test aux, not normalized
    def f_aux(params, batch):
        return f(params, batch), params

    f_aux_per_sample = torch.vmap(f_aux, in_dims=(None, 0))

    fisher_aux, _ = empirical_fisher(
        lambda p: f_aux_per_sample(p, batch), has_aux=True, normalize=False
    )(params)
    assert torch.allclose(fisher_aux, expected_fisher, rtol=1e-5)

    # Test aux, normalized
    fisher_aux_norm, _ = empirical_fisher(
        lambda p: f_aux_per_sample(p, batch), has_aux=True, normalize=True
    )(params)
    assert torch.allclose(fisher_aux_norm, expected_fisher / num_samples, rtol=1e-5)

    # Test matches fvp
    v = tree_map(lambda x: torch.randn_like(x), params)
    fvp_result = fvp(
        lambda p: f_per_sample(p, batch), (params,), (v,), normalize=False
    )[1]
    fvp_result = tree_ravel(fvp_result)[0]
    fisher_fvp = fisher @ tree_ravel(v)[0]
    assert torch.allclose(fvp_result, fisher_fvp, rtol=1e-5)


def test_diag_normal_log_prob():
    # Test tree mean and tree sd
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = {"a": torch.tensor([0.1, 0.2]), "b": torch.tensor([0.3, 0.4])}
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}

    mean_flat = torch.stack([mean["a"], mean["b"]]).flatten()
    sd_flat = torch.stack([sd_diag["a"], sd_diag["b"]]).flatten()
    x_flat = torch.stack([x["a"], x["b"]]).flatten()

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = torch.distributions.Normal(mean_flat, sd_flat).log_prob(x_flat).sum()

    assert torch.allclose(result, expected)

    # Test float mean and tree sd
    mean = 1.0
    mean_tree = tree_map(lambda t: torch.ones_like(t) * mean, x)

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = diag_normal_log_prob(x, mean_tree, sd_diag)

    assert torch.allclose(result, expected)

    # Test tree mean and float sd
    mean = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    sd_diag = 0.1

    sd_tree = tree_map(lambda t: torch.ones_like(t) * sd_diag, x)

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = diag_normal_log_prob(x, mean, sd_tree)

    assert torch.allclose(result, expected)

    # Test float mean and float sd
    mean = 1.0
    sd_diag = 0.1

    mean_tree = tree_map(lambda t: torch.ones_like(t) * mean, x)
    sd_tree = tree_map(lambda t: torch.ones_like(t) * sd_diag, x)

    result = diag_normal_log_prob(x, mean, sd_diag)
    expected = diag_normal_log_prob(x, mean_tree, sd_tree)

    assert torch.allclose(result, expected)

    # Test unnormalized
    result = diag_normal_log_prob(x, mean, sd_diag, normalize=False)
    expected = tree_reduce(
        torch.add,
        tree_map(
            lambda v, m, sd: -((v - m) ** 2 / (2 * sd**2)).sum(), x, mean_tree, sd_tree
        ),
    )
    assert torch.allclose(result, expected)

    # Test unnormalised gradient
    result_grad = torch.func.grad(diag_normal_log_prob)(x, mean, sd_diag)
    expected_grad = tree_map(lambda v, m, sd: -(v - m) / (sd**2), x, mean_tree, sd_tree)

    for key in result_grad.keys():
        assert torch.allclose(result_grad[key], expected_grad[key])


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

    # Test float sd_diag
    sd_diag = 0.1
    torch.manual_seed(42)
    result = diag_normal_sample(mean, sd_diag)
    torch.manual_seed(42)
    expected = tree_map(lambda m: torch.distributions.Normal(m, sd_diag).sample(), mean)

    for key in result:
        assert torch.equal(result[key], expected[key])


def test_tree_size():
    tree = {
        "a": 3.0,
        "b": torch.tensor([3.0, 4.0]),
    }

    result = tree_size(tree)
    expected = 3

    assert result == expected

    assert tree_size(1) == 1
    assert tree_size(1.2) == 1


def test_tree_extract():
    params = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
    }

    def check_func(x):
        return x[0] < 2.0

    result = tree_extract(check_func, params)

    expected = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([]),
    }

    for key in expected:
        assert torch.equal(result[key], expected[key])


def test_tree_insert():
    params = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
    }

    params_static = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
    }

    sub_params = {
        "a": torch.tensor([5.0, 6.0]),
        "b": torch.tensor([]),
    }

    def check_func(x):
        return x[0] < 2.0

    params2 = tree_insert(check_func, params, sub_params)

    expected = {
        "a": torch.tensor([5.0, 6.0]),
        "b": torch.tensor([3.0, 4.0]),
    }

    for key in expected:
        assert torch.equal(params2[key], expected[key])
        assert torch.equal(params[key], params_static[key])


def test_tree_insert_():
    params = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
    }

    sub_params = {
        "a": torch.tensor([5.0, 6.0]),
        "b": torch.tensor([]),
    }

    def check_func(x):
        return x[0] < 2.0

    params2 = tree_insert_(check_func, params, sub_params)

    expected = {
        "a": torch.tensor([5.0, 6.0]),
        "b": torch.tensor([3.0, 4.0]),
    }

    for key in expected:
        assert torch.equal(params2[key], expected[key])
        assert torch.equal(params[key], params2[key])


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


def test_tree_map_inplacify_():
    def func(x):
        return x + 1

    x = torch.tensor(1.0)
    y = tree_map_inplacify_(func, x)

    assert y == 2.0
    assert x == 2.0


def test_flexi_tree_map():
    def func(x):
        return x + 1

    x = torch.tensor(1.0)
    y = flexi_tree_map(func, x, inplace=False)

    assert y == 2.0
    assert x == 1.0

    y = flexi_tree_map(func, x, inplace=True)

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


def test_is_scalar():
    assert is_scalar(1)
    assert is_scalar(1.0)
    assert is_scalar(torch.ones(1))
    assert is_scalar(torch.tensor(1.0))
    assert is_scalar(torch.ones(1, 1))
    assert not is_scalar(torch.ones(2))
