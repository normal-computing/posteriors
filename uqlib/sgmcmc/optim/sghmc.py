from typing import Any
from torchopt.optim.base import Optimizer

from uqlib.sgmcmc import sghmc


class SGHMC(Optimizer):
    """SGHMC via PyTorch API.

    See Also:
        - The functional SGHMC sampler: :func:`uqlib.sgmcmc.sghmc`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Any,
        lr: float,
        alpha: float = 0.01,
        beta: float = 0.0,
    ) -> None:
        super().__init__(
            params,
            sghmc.build(
                lr=lr,
                alpha=alpha,
                beta=beta,
            ),
        )
