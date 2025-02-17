import torch
from tensordict import TensorClass, NonTensorData

from posteriors.tree_utils import tree_insert_


class TransformState(TensorClass["frozen"]):
    params: torch.Tensor
    aux: NonTensorData


def test_TransformState():
    s = TransformState(params=torch.ones(3), aux=None)

    params_new = torch.ones(3) * 2
    aux_new = ["dsadsa", torch.ones(3)]

    def update_state(state, params_new, aux_new):
        tree_insert_(state.params, params_new)
        # return state
        return state.replace(aux=NonTensorData(aux_new))

    state_new = update_state(s, params_new, aux_new)

    assert torch.allclose(state_new.params, params_new)
    assert state_new.aux == aux_new

    # Check params updated in-place
    assert torch.allclose(s.params, params_new)

    # aux never updated in-place as it is not guaranteed to be a TensorTree
    assert s.aux is None

    # Check works with vmap
    def init(params, aux=None):
        return TransformState(params=params, aux=aux)

    multi_params_new = torch.ones(5, 3)
    multi_aux_new = torch.ones(5, 3)

    multi_state_new = torch.vmap(init)(multi_params_new, multi_aux_new)

    assert torch.allclose(multi_state_new.params, multi_params_new)
    assert torch.allclose(multi_state_new.aux, multi_aux_new)
