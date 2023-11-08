import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


from uqlib import forward_multiple, diagonal_hessian


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def test_forward_multiple():
    model = TestModel()

    pvec = parameters_to_vector(model.parameters())

    pvec_multiple = torch.tile(pvec, (4, 1))

    input_single = torch.randn(10)
    input_multiple = torch.randn(8, 10)

    outputs_params_single_inputs_single = forward_multiple(model, pvec, input_single)
    assert outputs_params_single_inputs_single.shape == (1, 1, 1)

    outputs_params_multi_inputs_single = forward_multiple(
        model, pvec_multiple, input_single
    )
    assert outputs_params_multi_inputs_single.shape == (1, 4, 1)
    assert torch.allclose(
        outputs_params_multi_inputs_single, outputs_params_single_inputs_single
    )

    outputs_params_single_inputs_multi = forward_multiple(model, pvec, input_multiple)
    assert outputs_params_single_inputs_multi.shape == (8, 1, 1)

    outputs_params_multi_inputs_multi = forward_multiple(
        model, pvec_multiple, input_multiple
    )
    assert outputs_params_multi_inputs_multi.shape == (8, 4, 1)

    pvec_new = torch.randn_like(pvec)
    outputs_params_single_inputs_single_new = forward_multiple(
        model, pvec_new, input_single
    )
    assert outputs_params_single_inputs_single_new.shape == (1, 1, 1)
    assert not torch.equal(
        outputs_params_single_inputs_single_new, outputs_params_single_inputs_single
    )
    assert torch.equal(pvec, parameters_to_vector(model.parameters()))


def test_diagonal_hessian():
    f = lambda x, y: torch.sum(x**3 + y**2)
    diag_h = diagonal_hessian(f)

    x = torch.arange(10, dtype=torch.float32)

    y = 3.0

    assert torch.equal(diag_h(x, y), 6 * x)
