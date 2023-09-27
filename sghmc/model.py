from sghmc.modules.optimizer import init

import pytorch_lightning as pl


class SGHMCModel(pl.LightningModule):
    def __init__(self, model):
        self.automatic_optimization = False

        self.model = model
        self.optimizer = init()

    def training_step(self, batch):
        self.model(batch)
        return batch
