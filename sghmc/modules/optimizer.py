from typing import NamedTuple

import torch


class SGHMCState(NamedTuple):
    params: torch.LongTensor
    momenta: torch.LongTensor
    alpha: float
    beta: float


def init(
    params: torch.LongTensor,
    momenta: torch.Tensor = None,
    alpha: float = 0.01,
    beta: float = 0.0,
) -> None:
    if momenta is None:
        # momenta = torch.zeros_like(params)
        # momenta = [torch.zeros_like(p) for p in params]
        momenta = [torch.rand_like(p) for p in params]
    
    return SGHMCState(params, momenta, alpha, beta)


def step(state: SGHMCState, grad: torch.Tensor, stepsize: float) -> SGHMCState:
    params, momenta, alpha, beta = state
    
    params = [params[i] + stepsize * momenta[i] for i in range(len(params))]
    momenta = [momenta[i]
        -stepsize * grad[i]
        - stepsize * alpha * momenta
        + torch.sqrt(stepsize * (2 * alpha - stepsize * beta))
        * torch.randn_like(momenta)
    for i in range(len(params))]

    return SGHMCState(params, momenta, alpha, beta)
