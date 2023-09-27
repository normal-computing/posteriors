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
        momenta = torch.zeros_like(params)

    return SGHMCState(params, momenta, alpha, beta)


def step(state: SGHMCState, grad: torch.Tensor, stepsize: float) -> SGHMCState:
    params, momenta, alpha, beta = state

    params += stepsize * momenta
    momenta += (
        -stepsize * grad
        - stepsize * alpha * momenta
        + torch.sqrt(stepsize * (2 * alpha - stepsize * beta))
        * torch.randn_like(momenta)
    )

    return SGHMCState(params, momenta, alpha, beta)
