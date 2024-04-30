import torch

from posteriors import (
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
)


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
