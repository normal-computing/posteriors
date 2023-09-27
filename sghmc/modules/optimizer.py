from typing import NamedTuple, List

import torch


class SGHMCState(NamedTuple):
    params: List[torch.LongTensor]
    momenta: List[torch.LongTensor]
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
        momenta = [torch.rand_like(p).to(p) for p in params]

    return SGHMCState(params, momenta, alpha, beta)


def step(state: SGHMCState, grad: List[torch.Tensor], stepsize: float) -> SGHMCState:
    params, momenta, alpha, beta = state

    params = [params[i] + stepsize * momenta[i] for i in range(len(params))]
    momenta = [
        momenta[i]
        + stepsize * grad[i]
        - stepsize * alpha * momenta[i]
        + (stepsize * (2 * alpha - stepsize * beta)) ** 0.5
        * torch.randn_like(momenta[i]).to(momenta[i])
        for i in range(len(params))
    ]

    return SGHMCState(params, momenta, alpha, beta)
