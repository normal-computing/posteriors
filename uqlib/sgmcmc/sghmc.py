from typing import Any, Dict
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
        thinning=-1,
        momenta=None,
    ):
        params = list(params)
        super().__init__(params, {"lr": lr})

        self.current_step = 0
        self.thinning = thinning
        self.parameter_trajectory = {}

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

    def state_dict(self) -> Dict[str, Any]:
        opt_state_dict = super().state_dict()
        opt_state_dict["current_step"] = self.current_step
        return opt_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_step = state_dict.pop("current_step")
        return super().load_state_dict(state_dict)

    def save_params(self):
        if self.thinning > 0 and (self.current_step + 1) % self.thinning == 0:
            params_to_save = []
            for group in self.param_groups:
                params_to_save.append(
                    [p.detach().cpu().numpy() for p in group["params"]]
                )
            self.parameter_trajectory[self.current_step] = params_to_save

        self.current_step += 1

    def get_params(self):
        return self.parameter_trajectory

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

        self.save_params()

        return loss
