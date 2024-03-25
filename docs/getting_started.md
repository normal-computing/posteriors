## Installation
```
pip install uqlib
```

## Quick Start

`uqlib` is a Python library for uncertainty quantification and machine learning 
that is designed to be easy to use, flexible and extensible. It is built on top 
of [PyTorch](https://pytorch.org/docs/stable/index.html) and provides a range of 
tools for probabilistic modelling, Bayesian inference, and online learning.

`uqlib` algorithms conform to a very general structure

```py
transform = uqlib_algorithm.build(**config_args)

state = transform.init(params)

for batch in dataloader:
    state = transform.update(state, batch)
```
where

- `uqlib_algorithm` is a python module containing the `build`, `init`, and `update` 
functions that together define the `uqlib` algorithm.
- `build` is a function that loads `config_args` into the `init` and `update` functions
 and stores them within the `transform` instance.
- `init` constructs the iteration-varying `state` based on the model parameters `params`.
- `update` updates the `state` based on a new `batch` of data.

!!! example "I want more!"
    Our [API documentation](api/index.md) provide a detailed descriptions of the `uqlib`
    algorithms and utilities.

    Or check out our [examples](../examples) for some full walkthroughs!


## PyTrees

The internals of `uqlib` rely on [`optree`](https://optree.readthedocs.io/en/latest/) to
apply functions to arbitrary PyTrees of tensors. For example
```py
params_squared = optree.tree_map(lambda x: x**2, params)
```
will square all the tensors in the `params`, where `params` can be a 
`dict`, `list`, `tuple`, or any other [PyTree](https://github.com/metaopt/optree?tab=readme-ov-file#built-in-pytree-node-types).

`uqlib` also provides a [`uqlib.flexi_tree_map`][] function that allows for in-place support
```py
params_squared = optree.flexi_tree_map(lambda x: x**2, params, inplace=True)
```
Here the tensors of params are modified in-place, without assigning extra memory.


## [`torch.func`](https://pytorch.org/docs/stable/func.html)

Instead of using `torch`'s more common `loss.backward()` style automatic differentiation,
`uqlib` uses a functional approach, via `torch.func.grad` and friends. The functional 
approach is easier to test, composes better with other tools and importantly for `uqlib` 
it makes for code that is closer to the mathematical notation.

For example, the gradient of a function `f` with respect to `x` can be computed as
```py
grad_f_x = torch.func.grad(f)(x)
```
where `f` is a function that takes `x` as input and returns a scalar output. Again, 
`x` can be a `dict`, `list`, `tuple`, or any other PyTree with `torch.Tensor` leaves.
