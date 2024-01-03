from typing import Any
from torchopt.optim.base import Optimizer

from uqlib.sgmcmc import sghmc


class SGHMC(Optimizer):
    """SGHMC via PyTorch API.

    See Also:
        - The functional SGHMC sampler in `uqlib.sgmcmc.sghmc`.
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        alpha: float = 0.01,
        beta: float = 0.0,
        maximize: bool = True,
    ) -> None:
        """Initialise SGHMC.

        Args:
            params: Parameters to optimise.
            lr: Learning rate.
            alpha: Friction coefficient.
            beta: Gradient noise coefficient (estimated variance).
            maximize: Whether to maximize (ascend) or minimise (descend).
        """
        super().__init__(
            params,
            sghmc.build(
                lr=lr,
                alpha=alpha,
                beta=beta,
                maximize=maximize,
            ),
        )
