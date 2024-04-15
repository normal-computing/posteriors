import os
from torch import nn, utils
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torchopt
from dataclasses import asdict

import posteriors

# Example from https://lightning.ai/docs/pytorch/stable/starter/introduction.html

method, config_args = posteriors.vi.diag, {"optimizer": torchopt.adam(lr=1e-3)}
# method, config_args = posteriors.sgmcmc.sghmc, {"lr": 1e-3}

encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

encoder_function = posteriors.model_to_function(encoder)
decoder_function = posteriors.model_to_function(decoder)


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


# define the LightningModule
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
