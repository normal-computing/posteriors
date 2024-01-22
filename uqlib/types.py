from typing import TypeAlias
from optree.typing import PyTreeTypeVar
from torch import Tensor

TensorTree: TypeAlias = PyTreeTypeVar("TensorTree", Tensor)
