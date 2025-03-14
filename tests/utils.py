from typing import Any
import torch
from optree import tree_leaves
from posteriors.types import Transform, TensorTree


def kl_gaussians(
    m0: torch.Tensor, c0: torch.Tensor, m1: torch.Tensor, c1: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (
        torch.trace(c1.inverse() @ c0)
        + torch.dot(m1 - m0, c1.inverse() @ (m1 - m0))
        - len(m0)
        + torch.log(torch.linalg.det(c1) / torch.linalg.det(c0))
    )


def verify_inplace_update(transform: Transform, params: TensorTree, batch: Any):
    init_state = transform.init(params)

    # One step not inplace
    torch.manual_seed(42)
    state_not_inplace, _ = transform.update(init_state, batch)

    # One step in place = same as not inplace
    torch.manual_seed(42)
    state_inplace, _ = transform.update(init_state, batch, inplace=True)

    # Iterate through leaves of params and check they are the same
    for init_leaf, not_inplace_leaf, inplace_leaf in zip(
        tree_leaves(init_state.to_dict()),
        tree_leaves(state_not_inplace.to_dict()),
        tree_leaves(state_inplace.to_dict()),
    ):
        assert torch.equal(inplace_leaf, not_inplace_leaf)
        assert torch.equal(init_leaf, not_inplace_leaf)
