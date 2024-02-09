import torch

import uqlib


def test_optim():
    optimizer_cls = torch.optim.SGD
    lr = 0.1

    def loss_fn(p, b):
        return torch.sum(p**2), torch.tensor([])

    transform = uqlib.optim.build(optimizer_cls, loss_fn, lr=lr)

    params = torch.tensor([1.0], requires_grad=True)
    state = transform.init(params)

    for _ in range(100):
        state = transform.update(state, torch.tensor([1.0]))

    assert state.loss < 1e-3
    assert state.params < 1e-3
