from typing import NamedTuple, List

import torch
import torch.nn as nn
from torch.optim import Optimizer


class SGHMC(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 0.01,
        beta: float = 0.0,
        momenta=None,
    ):
        params = list(params)
        super().__init__(params, {"lr": lr})

        self.alpha = alpha
        self.beta = beta

        if momenta is None:
            momenta = [torch.zeros_like(p).to(p) for p in params]

        for group in self.param_groups:
            group["momenta"] = [
                nn.Parameter(torch.zeros_like(p), requires_grad=False).to(p)
                for p in group["params"]
            ]

    def step(self, closure=None):
        # Perform the optimization step for each parameter group
        for group in self.param_groups:
            # Iterate over the parameters in the current group
            for p, r in zip(group["params"], group["momenta"]):
                if p.grad is None:
                    continue

                # Update the parameter using the SGD update rule
                lr = group["lr"]
                p.data += lr * r.data
                r.data = (
                    r.data
                    - lr * p.grad
                    - lr * self.alpha * r.data
                    + (lr * (2 * self.alpha - lr * self.beta)) ** 0.5
                    * torch.randn_like(r.data)
                )

                # Zero out the gradient to avoid accumulation
                p.grad.data.zero_()


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
        momenta = [torch.zeros_like(p).to(p) for p in params]
        # momenta = [torch.rand_like(p).to(p) for p in params]

    return SGHMCState(params, momenta, alpha, beta)


def step(state: SGHMCState, grad: List[torch.Tensor], stepsize: float) -> SGHMCState:
    params, momenta, alpha, beta = state

    new_params = [params[i] + stepsize * momenta[i] for i in range(len(params))]
    new_momenta = [
        momenta[i]
        - stepsize * grad[i]
        - stepsize * alpha * momenta[i]
        + (stepsize * (2 * alpha - stepsize * beta)) ** 0.5
        * torch.randn_like(momenta[i]).to(momenta[i])
        for i in range(len(params))
    ]

    # params = [params[i] - stepsize * grad[i] for i in range(len(params))]

    return SGHMCState(new_params, new_momenta, alpha, beta)


if __name__ == "__main__":
    import torch.nn as nn

    model = nn.Linear(1, 1)
    print(list(model.parameters())[0])
    optimizer = SGHMC(model.parameters())
    print(optimizer.param_groups)
