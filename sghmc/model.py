from typing import Any, Dict
from .optimizer.optimizer import SGHMC

import pytorch_lightning as pl
import torch.nn.functional as F


class SGHMCModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.automatic_optimization = False

        self.save_parameters = []

        self.model = model
        self.lr = lr

    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y.squeeze(-1))
        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["save_params"] = self.save_parameters
        return super().on_save_checkpoint(checkpoint)

    def on_after_backward(self) -> None:
        params = [p.detach().cpu().numpy() for p in list(self.model.parameters())]
        self.save_parameters = self.save_parameters + [params]
        return super().on_after_backward()

    def configure_optimizers(self):
        return SGHMC(self.model.parameters(), lr=self.lr)
