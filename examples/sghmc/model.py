from typing import Any, Dict
import pytorch_lightning as pl
import torch.nn.functional as F

from uqlib.optimizers.sghmc import SGHMC


class SGHMCModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3, alpha=1e-1, beta=1e-1, thinning=5):
        super().__init__()
        self.model = model

        self.save_hyperparameters(
            {"learning_rate": lr, "alpha": alpha, "beta": beta, "thinning": thinning}
        )

        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.thinning = thinning

    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y.squeeze(-1))

        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        trajectory_params = self.optimizers().get_params()
        checkpoint["save_params"] = trajectory_params
        return super().on_save_checkpoint(checkpoint)

    def configure_optimizers(self):
        return SGHMC(
            self.model.parameters(),
            lr=self.lr,
            alpha=self.alpha,
            beta=self.beta,
            thinning=self.thinning,
        )
