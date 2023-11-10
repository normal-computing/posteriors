import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


from uqlib import dict_map, forward_multiple, diagonal_hessian


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def test_dict_map():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 3, "b": 4}
    result = dict_map(lambda x, y: x + y, d1, d2)
    assert result == {"a": 4, "b": 6}

    d1 = {"a": 10, "b": [1, 2]}
    d2 = {"a": 20, "b": [3, 4]}
    result = dict_map(lambda x, y: x + y, d1, d2)
    assert result == {"a": 30, "b": [1, 2, 3, 4]}

    d1 = {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}
    d2 = {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])}
    result = dict_map(torch.add, d1, d2)
    expected = {"a": torch.tensor([6, 8]), "b": torch.tensor([10, 12])}
    for key in result:
        assert torch.equal(result[key], expected[key])


def test_forward_multiple():
    model = TestModel()

    pvec = parameters_to_vector(model.parameters())

    pvec_multiple = torch.tile(pvec, (4, 1))

    input_single = torch.randn(10)
    input_multiple = torch.randn(8, 10)

    outputs_params_single_inputs_single = forward_multiple(model, input_single, pvec)
    assert outputs_params_single_inputs_single.shape == (1, 1, 1)

    outputs_params_multi_inputs_single = forward_multiple(
        model, input_single, pvec_multiple
    )
    assert outputs_params_multi_inputs_single.shape == (1, 4, 1)
    assert torch.allclose(
        outputs_params_multi_inputs_single, outputs_params_single_inputs_single
    )

    outputs_params_single_inputs_multi = forward_multiple(model, input_multiple, pvec)
    assert outputs_params_single_inputs_multi.shape == (8, 1, 1)

    outputs_params_multi_inputs_multi = forward_multiple(
        model, input_multiple, pvec_multiple
    )
    assert outputs_params_multi_inputs_multi.shape == (8, 4, 1)

    pvec_new = torch.randn_like(pvec)
    outputs_params_single_inputs_single_new = forward_multiple(
        model, input_single, pvec_new
    )
    assert outputs_params_single_inputs_single_new.shape == (1, 1, 1)
    assert not torch.equal(
        outputs_params_single_inputs_single_new, outputs_params_single_inputs_single
    )
    assert torch.equal(pvec, parameters_to_vector(model.parameters()))


def test_diagonal_hessian():
    # Test with a constant function
    def const_fn(_):
        return torch.tensor(3.0)

    hessian_diag = diagonal_hessian(const_fn)
    x = torch.tensor([1.0, 2.0])
    result = hessian_diag(x)
    expected = torch.zeros_like(x)
    assert torch.equal(result, expected)

    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    result = hessian_diag(x)
    expected = dict_map(lambda v: torch.zeros_like(v), x)
    for key in result:
        assert torch.equal(result[key], expected[key])

    # Test with a linear function
    def linear_fn(x):
        return x["a"].sum() + x["b"].sum()

    hessian_diag = diagonal_hessian(linear_fn)
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    result = hessian_diag(x)
    for key in result:
        assert torch.equal(result[key], expected[key])

    # Test with a quadratic function
    def quad_fn(x):
        return (x["a"] ** 2).sum() + (x["b"] ** 2).sum()

    hessian_diag = diagonal_hessian(quad_fn)
    x = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    result = hessian_diag(x)
    expected = dict_map(lambda v: 2 * torch.ones_like(v), x)
    for key in result:
        assert torch.equal(result[key], expected[key])
