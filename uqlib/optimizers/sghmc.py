from torch.optim import Optimizer
import torch.nn as nn
import torch


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
            momenta = [torch.zeros_like(p) for p in params]

        for group in self.param_groups:
            group["momenta"] = []
            for p in group["params"]:
                group["momenta"].append(
                    nn.Parameter(torch.zeros_like(p), requires_grad=False)
                )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

        return loss
