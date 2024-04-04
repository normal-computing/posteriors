# Autoencoder with Lightning

In this example, we'll adapt the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
to use UQ methods with `posteriors` and logging + device handling with `lightning`.



## PyTorch model
We begin by defining the PyTorch model. This is unchanged from the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html):

```python
import os
from torch import nn, utils
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torchopt
from dataclasses import asdict

import posteriors

encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

encoder_function = posteriors.model_to_function(encoder)
decoder_function = posteriors.model_to_function(decoder)
```


## Log posterior

As mentioned in the [constructing log posteriors](../log_posteriors.md) page,
the `log_posterior` function depends on the amount of data we have in the training set,
`num_data`. We don't know that yet so we'll define it later.

Otherwise, the negative log likelihood is the same reconstruction loss as in the
[Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html):

```python
def log_posterior(params, batch, num_data):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = encoder_function(params[0], x)
    x_hat = decoder_function(params[1], z)
    log_lik = (
        torch.distributions.Normal(x_hat, 1, validate_args=False)
        .log_prob(x)
        .sum(-1)
        .mean()
    )
    log_prior = posteriors.diag_normal_log_prob(
        params[0]
    ) + posteriors.diag_normal_log_prob(params[1])
    return log_lik + log_prior / num_data, x_hat
```


## `posteriors` method

We can now define the `posteriors` method. For example, we could us [`vi.diag`](../api/vi/diag.md)
```python
method, config_args = posteriors.vi.diag, {"optimizer": torchopt.adam(lr=1e-3)}
```
or [`sgmcmc.sghmc`](../api/sgmcmc/sghmc.md):
```python
method, config_args = posteriors.sgmcmc.sghmc, {"lr": 1e-3}
```
We can easily swap methods using `posteriors`'s unified interface.


## Lightning module

The `LightningModule` is the same as in the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html),
with a few minor modifications:

- We add a `num_data` attribute to the class, as well as a `log_posterior` method that
depends on it.
- The training step now simply calls `update` and logs the float attributes.
- We use `configure_optimizers` to build the `transform` object and the `state`.
We do not return optimizers, as we do not want lightning's automatic optimization.
- We load the number of data points in the training set in `on_train_start`, when
the module has `train_dataloader` available.
- We save and load the state of the transform in `on_save_checkpoint` and
`on_load_checkpoint` so that we can resume training if needed.

```python
class LitAutoEncoderUQ(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_data = None

    def log_posterior(self, params, batch):
        return log_posterior(params, batch, self.num_data)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        self.state = self.transform.update(self.state, batch, inplace=True)
        # Logging to TensorBoard (if installed) by default
        for k, v in asdict(self.state).items():
            if isinstance(v, float) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                self.log(k, v)

    def configure_optimizers(self):
        # We don't need to return optimizers here, as we are using the `transform` object
        # rather than lightning's automatic optimization
        self.transform = method.build(self.log_posterior, **config_args)
        all_params = [
            dict(self.encoder.named_parameters()),
            dict(self.decoder.named_parameters()),
        ]
        self.state = self.transform.init(all_params)

    def on_train_start(self) -> None:
        # Load the number of data points used for log_posterior
        self.num_data = len(self.trainer.train_dataloader.dataset)

    def on_save_checkpoint(self, checkpoint):
        # Save the state of the transform
        checkpoint["state"] = self.state

    def on_load_checkpoint(self, checkpoint):
        # Load the state of the transform
        self.state = checkpoint["state"]
```


## Load dataset and train!

Just as in the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html):

```python
# init the autoencoder
autoencoderuq = LitAutoEncoderUQ(encoder, decoder)

# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# train the model
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoderuq, train_dataloaders=train_loader)
```