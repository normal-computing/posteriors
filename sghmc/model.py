from typing import Any, Dict
from .optimizer.optimizer import SGHMC

import pytorch_lightning as pl
import torch.nn.functional as F


class SGHMCModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3, alpha=1e-1, beta=1e-1):
        super().__init__()
        self.automatic_optimization = False

        self.model = model

        self.hparams = {"learning_rate": lr, "alpha": alpha, "beta": beta}
        self.save_parameters_trajectory = []

        self.lr = lr
        self.alpha = alpha
        self.beta = beta

    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y.squeeze(-1))

        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["save_params"] = self.save_parameters_trajectory
        return super().on_save_checkpoint(checkpoint)

    def on_after_backward(self) -> None:
        params = [p.detach().cpu().numpy() for p in list(self.model.parameters())]
        self.save_parameters_trajectory = self.save_parameters_trajectory + [params]
        return super().on_after_backward()

    def configure_optimizers(self):
        return SGHMC(self.model.parameters(), lr=self.lr)
