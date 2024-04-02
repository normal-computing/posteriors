<div align="center">
<img src="https://storage.googleapis.com/posteriors/logo_with_text.png" alt="logo"></img>
</div>

[**Friends**](#friends)
| [**Contributing**](#contributing)
| [**Documentation**](https://normal-computing.github.io/posteriors/)

## What is `posteriors`?

General purpose python library for **U**ncertainy **Q**uantification with [`PyTorch`](https://github.com/pytorch/pytorch).

`posteriors` is functional first and aims to be easy to use and extend. Iterative `posteriors` algorithms take the following unified form
```python
# Load model
# Load dataloader
# Define config arguments for chosen algorithm
transform = algorithm.build(**config)
state = transform.init(dict(model.named_parameters()))

for batch in dataloader:
    state = transform.update(state, batch)
```

`transform` is an algorithm kernel that is pre-built with all the necessary configuration arguments. For example:
```python
num_data = len(dataloader.dataset)
functional_model = posteriors.model_to_function(model)

def log_posterior(params, batch):
    predictions = functional_model(params, batch)
    log_posterior = -loss_fn(predictions, batch) + prior(params) / num_data
    return log_posterior, predictions

optimizer = torchopt.adam(lr=1e-3)
transform = posteriors.vi.diag.build(log_posterior, optimizer, temperature=1/num_data)
```

Observe that `posteriors` recommends specifying `log_posterior` and `temperature` such that 
`log_posterior` remains on the same scale for different batch sizes. `posteriors` 
algorithms are designed to be stable as `temperature` goes to zero.

Further the output of `log_posterior` is a tuple containing the evaluation 
(single-element Tensor) and an additional argument (TensorTree) containing any 
auxiliary information we'd like to retain from the model call, here the model predictions.
If you have no auxiliary information, you can simply return `torch.tensor([])` as
the second element. For more info see [`torch.func.grad`](https://pytorch.org/docs/stable/generated/torch.func.grad.html) 
(with `has_aux=True`) or the [posteriors documentation](https://normal-computing.github.io/posteriors/log_posteriors).

Check out the [tutorials](https://normal-computing.github.io/posteriors/tutorials) for more detailed usage!

## Friends

Interfaces seamlessly with:

- [`torch`](https://github.com/pytorch/pytorch) and in particular [`torch.func`](https://pytorch.org/docs/stable/func.html).
- [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) for distributions and sampling, (note that it's typically required to set `validate_args=False` to conform with the control flows in [`torch.func`](https://pytorch.org/docs/stable/func.html)).
- Functional and flexible torch optimizers from [`torchopt`](https://github.com/metaopt/torchopt), 
    (which is the default for [`posteriors.vi`](posteriors/vi/) but `torch.optim` also interfaces easily).
- [`transformers`](https://github.com/huggingface/transformers) for pre-trained models.
- [`lightning`](https://github.com/Lightning-AI/lightning) for convenient training and logging, see [examples/lightning_autoencoder.py](examples/lightning_autoencoder.py).

The functional transform interface is strongly inspired by frameworks such as 
[`optax`](https://github.com/google-deepmind/optax) and [`BlackJAX`](https://github.com/blackjax-devs/blackjax).


## Contributing

You can report a bug or request a feature by [creating a new issue on GitHub](https://github.com/normal-computing/posteriors/issues).

If you want to contribute code, please check the [contributing guide](https://normal-computing.github.io/posteriors/contributing).
