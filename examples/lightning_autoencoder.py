from functools import partial
import os
from torch import nn, utils
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torchopt
import optree

import uqlib

# Example from https://lightning.ai/docs/pytorch/stable/starter/introduction.html

method, config_args = uqlib.vi.diag, {"optimizer": torchopt.adam(lr=1e-3)}
# method, config_args = uqlib.sgmcmc.sghmc, {"lr": 1e-3}

encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

encoder_function = uqlib.model_to_function(encoder)
decoder_function = uqlib.model_to_function(decoder)


def log_normal_prior(params):
    all_vals = optree.tree_map(
        lambda p: torch.distributions.Normal(0, 1, validate_args=False)
        .log_prob(p)
        .sum(),
        params,
    )
    return optree.tree_reduce(torch.add, all_vals)


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
    log_prior = log_normal_prior(params[0]) + log_normal_prior(params[1])
    return log_lik + log_prior / num_data, x_hat


# define the LightningModule
class LitAutoEncoderUQ(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        self.state = self.transform.update(self.state, batch, inplace=True)
        # Logging to TensorBoard (if installed) by default
        for k, v in self.state._asdict().items():
            if isinstance(v, float):
                self.log(k, v)

    def configure_optimizers(self):
        pass

    def on_train_start(self) -> None:
        self.log_posterior = partial(
            log_posterior, num_data=len(self.trainer.train_dataloader.dataset)
        )
        self.transform = method.build(self.log_posterior, **config_args)
        all_params = [
            dict(self.encoder.named_parameters()),
            dict(self.decoder.named_parameters()),
        ]
        self.state = self.transform.init(all_params)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state"] = self.state

    def on_load_checkpoint(self, checkpoint):
        self.state = checkpoint["state"]


autoencoderuq = LitAutoEncoderUQ(encoder, decoder)


# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# train the model
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoderuq, train_dataloaders=train_loader)


checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoderUQ.load_from_checkpoint(
    checkpoint, encoder=encoder, decoder=decoder
)


assert hasattr(autoencoder, "state")
