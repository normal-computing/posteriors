from sghmc.modules.optimizer import init, step

import pytorch_lightning as pl
import torch.nn.functional as F
import torch


class SGHMCModel(pl.LightningModule):
    def __init__(self, model, stepsize, init_params=None):
        super().__init__()
        self.automatic_optimization = False

        self.stepsize = stepsize

        self.model = model

        if not init_params:
            init_params = [torch.zeros_like(p) for p in self.model.parameters()]

        self.optimizer_state = init(init_params)
        self.all_params = self.optimizer_state.params

    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y)

        self.manual_backward(loss)

        new_state = step(self.optimizer_state, self.model.grad, self.stepsize)
        self.all_params = torch.stack([self.all_params, new_state.params])
