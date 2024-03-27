import torch
import torchopt

import posteriors


def test_torchopt_sgd():
    optimizer = torchopt.sgd(lr=0.1)

    def loss_fn(p, b):
        return torch.sum(p**2), torch.tensor([])

    transform = posteriors.torchopt.build(loss_fn, optimizer)

    # Test inplace=False
    params = torch.tensor([1.0], requires_grad=True)
    state = transform.init(params)

    for _ in range(200):
        state = transform.update(state, torch.tensor([1.0]), inplace=False)

    assert state.loss < 1e-3
    assert state.params < 1e-3
    assert not torch.allclose(params, state.params)

    # Test inplace=True
    params = torch.tensor([1.0], requires_grad=True)
    state = transform.init(params)

    for _ in range(200):
        state = transform.update(state, torch.tensor([1.0]), inplace=True)

    assert state.loss < 1e-3
    assert state.params < 1e-3
    assert torch.allclose(params, state.params)


def test_torchopt_adamw():
    optimizer = torchopt.adamw(lr=0.1)

    def loss_fn(p, b):
        return torch.sum(p**2), torch.tensor([])

    transform = posteriors.torchopt.build(loss_fn, optimizer)

    # Test inplace=False
    params = torch.tensor([1.0], requires_grad=True)
    state = transform.init(params)

    for _ in range(200):
        state = transform.update(state, torch.tensor([1.0]), inplace=False)

    assert state.loss < 1e-3
    assert state.params < 1e-3
    assert not torch.allclose(params, state.params)

    # Test inplace=True
    params = torch.tensor([1.0], requires_grad=True)
    state = transform.init(params)

    for _ in range(200):
        state = transform.update(state, torch.tensor([1.0]), inplace=True)

    assert state.loss < 1e-3
    assert state.params < 1e-3
    assert torch.allclose(params, state.params)
