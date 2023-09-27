from typing import NamedTuple, List

import torch
from torch.optim import Optimizer


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


class SGHMC(Optimizer):
    """Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
    procedure to adapt its own hyperparameters during the initial stages
    of sampling.

    See [1] for more details on this burn-in procedure.\n
    See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

    [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
        In Advances in Neural Information Processing Systems 29 (2016).\n
        `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
    [2] T. Chen, E. B. Fox, C. Guestrin
        In Proceedings of Machine Learning Research 32 (2014).\n
        `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """

    name = "SGHMC"

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        num_burn_in_steps: int = 3000,
        noise: float = 0.0,
        mdecay: float = 0.05,
        scale_grad: float = 1.0,
    ) -> None:
        """Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        num_burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Set to `0` to turn scale adaption off.
            Default: `3000`.
        noise: float, optional
            (Constant) per-parameter noise level.
            Default: `0.`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr,
            scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            noise=noise,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, noise, lr = group["mdecay"], group["noise"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"])

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1.0 / (tau + 1.0)
                minv_t = 1.0 / torch.sqrt(v_hat)

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1.0 - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient**2))
                #  }}} Burn-in updates #

                lr_scaled = lr / torch.sqrt(scale_grad)

                #  Draw random sample {{{ #

                noise_scale = (
                    2.0 * (lr_scaled**2) * mdecay * minv_t
                    - 2.0 * (lr_scaled**3) * (minv_t**2) * noise
                    - (lr_scaled**4)
                )

                sigma = torch.sqrt(torch.clamp(noise_scale, min=1e-16))

                # sample_t = torch.normal(mean=0., std=torch.tensor(1.)) * sigma
                sample_t = torch.normal(mean=0.0, std=sigma)
                #  }}} Draw random sample #

                #  SGHMC Update {{{ #
                momentum_t = momentum.add_(
                    -(lr**2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum_t)
                #  }}} SGHMC Update #

        return loss
