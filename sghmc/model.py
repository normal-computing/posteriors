from .optimizer.optimizer import SGHMC

import pytorch_lightning as pl
import torch.nn.functional as F


class SGHMCModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.automatic_optimization = False

        self.model = model
        self.lr = lr

    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y.squeeze(-1))
        return loss

    def configure_optimizers(self):
        return SGHMC(self.model.parameters(), lr=self.lr)
