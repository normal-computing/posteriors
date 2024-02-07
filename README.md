# uqlib


General purpose python library for **U**ncertainy **Q**uantification with [`torch`](https://github.com/pytorch/pytorch).

`uqlib` is functional first and aims to be easy to use and extend. Iterative `uqlib` algorithms take the following unified form
```python
state = transform.init(dict(model.named_parameters()))

for batch in dataloader:
    state = transform.update(state, batch)
```

`transform` is an algorithm kernel that is pre-built with all the necessary configuration arguments. For example:
```python
num_data = len(dataloader.dataset)
functional_model = uqlib.model_to_function(model)

def log_posterior(params, batch):
    predictions = functional_model(params, batch)
    log_posterior = -loss_fn(predictions, batch) + prior(params) / num_data
    return log_posterior, predictions

optimizer = partial(torchopt.Adam, lr=1e-3)
transform = uqlib.vi.diag.build(log_posterior, optimizer, temperature=1/num_data)
```

Observe that `uqlib` recommends specifying `log_posterior` and `temperature` such that 
`log_posterior` remains on the same scale for different batch sizes. `uqlib` 
algorithms are designed to be stable as `temperature` goes to zero.

Further the output of `log_posterior` is a tuple containing the evaluation 
(single-element Tensor) and an additional argument (TensorTree) containing any 
auxiliary information we'd like to retain from the model call, here the model predictions.
If you have no auxiliary information, you can simply return `torch.tensor([])` as
the second element. For more info see e.g. [`torch.func.grad`](https://pytorch.org/docs/stable/generated/torch.func.grad.html) 
(with `has_aux=True`).

## Friends

Interfaces seamlessly with:

- [`torch`](https://github.com/pytorch/pytorch) and in particular [`torch.func`](https://pytorch.org/docs/stable/func.html).
- [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) for distributions and sampling, (note that it's typically required to set `validate_args=False` to conform with the control flows in [`torch.func`](https://pytorch.org/docs/stable/func.html)).
- Functional and flexible torch optimizers from [`torchopt`](https://github.com/metaopt/torchopt), 
    (which is the default for [`uqlib.vi`](uqlib/vi/) but `torch.optim` also interfaces easily).
- [`transformers`](https://github.com/huggingface/transformers) for pre-trained models.
- [`lightning`](https://github.com/Lightning-AI/lightning) for convenient training and logging, see [examples/lightning_autoencoder.py](examples/lightning_autoencoder.py).

The functional transform interface is strongly inspired by frameworks such as 
[`optax`](https://github.com/google-deepmind/optax) and [`BlackJAX`](https://github.com/blackjax-devs/blackjax).


## Contributing

You can report a bug or request a feature by [creating a new issue on GitHub](https://github.com/normal-computing/uqlib/issues).

Pull requests are welcomed! Please go through the following steps:

1. Create a new branch from `main`.
2. Run `pip install -e .` to install the package in editable mode.
3. Add your code and tests (`tests` has the same structure as `uqlib`).
4. Run `pre-commit run --all-files` to check your code lints
(or your can run `pre-commit install` in the repo to make `pre-commit` 
run automatically on `git commit`).
5. Run `python -m pytest` to check that all tests pass.
6. Commit your changes and push your branch to GitHub.
7. Create pull request into the `main` branch.

Feel free to open a draft PR to discuss changes or get feedback.

