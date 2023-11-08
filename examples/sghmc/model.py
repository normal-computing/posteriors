from typing import Any, Dict
import pytorch_lightning as pl
import torch.nn.functional as F

from uqlib.optimizers.sghmc import SGHMC


class SGHMCModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3, alpha=1e-1, beta=1e-1):
        super().__init__()
        self.model = model

        self.save_hyperparameters({"learning_rate": lr, "alpha": alpha, "beta": beta})
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

    # def on_train_batch_end(self, *args, **kwargs) -> None:
    #     params = [p.detach().cpu().numpy() for p in list(self.model.parameters())]
    #     self.save_parameters_trajectory = self.save_parameters_trajectory + [params]
    #     return super().on_train_batch_end(*args, **kwargs)

    def configure_optimizers(self):
        return SGHMC(
            self.model.parameters(), lr=self.lr, alpha=self.alpha, beta=self.beta
        )
