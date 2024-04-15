import torch
from dataclasses import dataclass
import optree

from posteriors.types import TransformState


def test_TransformState():
    @dataclass
    class s(TransformState):
        params: torch.Tensor
        aux: torch.Tensor
        b: dict

    a = s(torch.ones(3), aux=torch.ones(3), b={"tens": torch.ones(3)})

    children, metadata = optree.tree_flatten(a)

    assert len(children) == 3
    assert len(metadata) == 3

    a2 = optree.tree_unflatten(metadata, children)
    children2, metadata2 = optree.tree_flatten(a2)

    for i, j in zip(children, children2):
        assert torch.allclose(i, j)

    y = optree.tree_map(lambda x: x * 2, a)
    children_y, metadata_y = optree.tree_flatten(y)

    for i, j in zip(children, children_y):
        assert torch.allclose(i * 2, j)
