## `torch.no_grad`

If you find yourself running out of memory when using `torch.func.grad` and friends,
it might be because `torch` is trying to accumulate gradients through your
`torch.func.grad`calls. To prevent this, somewhat counterintuitively, 
wrap your code in `torch.no_grad`:

```python
with torch.no_grad():
    grad_f_x = torch.func.grad(f)(params, batch)
```

Don't worry, `torch.no_grad` won't prevent the gradients being calculated correctly
in the functional call. However, `torch.inference_mode` will turn autograd off
altogether. More info in the [torch.func docs](https://pytorch.org/docs/stable/generated/torch.func.grad.html).


## `validate_args=False` in `torch.distributions`

`posteriors` uses `torch.vmap` internally to vectorize over functions, for cool things like
[per-sample gradients](https://pytorch.org/tutorials/intermediate/per_sample_grads.html).
The `validate_args=True` control flows in `torch.distributions` do not compose with the 
control flows in `torch.vmap`. So it is recommended to set `validate_args=False` when 
using `torch.distributions` in `posteriors`:

```python
import torch
from torch.distributions import Normal

torch.vmap(lambda x: Normal(0., 1.).log_prob(x))(torch.arange(3))
# RuntimeError: vmap: It looks like you're attempting to use a
# Tensor in some data-dependent control flow. We don't support that yet, 
# please shout over at https://github.com/pytorch/functorch/issues/257 .

torch.vmap(lambda x: Normal(0., 1., validate_args=False).log_prob(x))(torch.arange(3))
# tensor([-0.9189, -1.4189, -2.9189])
```

## Auxiliary information

`posteriors` enforces `log_posterior` and `log_likelihood` functions to have a
`log_posterior(params, batch) -> log_prob, aux` signature, where the second element
contains any auxiliary information. If you don't have any auxiliary information, just
return an empty tensor:

```python
def log_posterior(params, batch):
    log_prob = ...
    return log_prob, torch.tensor([])
```

More info in the [constructing log posteriors](log_posteriors.md) page.


## `inplace`

All `posteriors` algorithms have an `update` function with signature
`update(state, batch, inplace=False) -> state`[^1]. The `inplace`
argument can be set to `True` to update the `state` in-place and save memory. However,
`posteriors` is functional first, so has `inplace=False` as the default. 

[^1]: Assuming all other args and kwargs have been pre-configured with by the `build` function


```python
state2 = transform.update(state, batch)
# state is not updated

state2 = transform.update(state, batch, inplace=True)
# state is updated and state2 is a pointer to state
```


## `torch.tensor` with autograd

As specified in the [documentation](https://pytorch.org/docs/stable/generated/torch.tensor.html),
`torch.tensor` does not preserve autograd history. If you want to construct a tensor
within a differentiable function, use [`torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html) instead:

```python
def f_with_tensor(x):
    return torch.tensor([x**2, x**3]).sum()

torch.func.grad(f_with_tensor)(torch.tensor(2.))
# tensor(0.)

def f_with_stack(x):
    return torch.stack([x**2, x**3]).sum()

torch.func.grad(f_with_stack)(torch.tensor(2.))
# tensor(16.)
```
