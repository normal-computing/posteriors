from collections import defaultdict
from typing import Any, DefaultDict
from torchopt.optim.base import Optimizer

from uqlib.sgmcmc import sghmc


class SGHMC(Optimizer):
    """SGHMC via PyTorch API.

    See Also:
        - The functional SGHMC sampler in `uqlib.sgmcmc.sghmc`.
    """

    defaults: dict = {}
    state: DefaultDict = defaultdict()

    def __init__(
        self,
        params: Any,
        lr: float,
        alpha: float = 0.01,
        beta: float = 0.0,
        temperature: float = 1.0,
        maximize: bool = True,
        momenta: Any | None = None,
    ) -> None:
        """Initialise SGHMC.

        Args:
            params: Parameters to optimise.
            lr: Learning rate.
            alpha: Friction coefficient.
            beta: Gradient noise coefficient (estimated variance).
            temperature: Temperature of the joint parameter + momenta distribution.
            maximize: Whether to maximize (ascend) or minimise (descend).
            momenta: Initial momenta. Defaults to all zeroes.
        """
        super().__init__(
            params,
            sghmc.build(
                lr=lr,
                alpha=alpha,
                beta=beta,
                temperature=temperature,
                maximize=maximize,
                momenta=momenta,
            ),
        )
