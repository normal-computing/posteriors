import os
from examples.lora_transformer import TransformerModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from experiments.utils.utils import save_config
from ml_collections.config_dict import FrozenConfigDict
from experiments.base import Experiment
from experiments.laplace_lora.dataset import HuggingfaceDataset
import wandb


class LoRAExperiment(Experiment):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)
        self.test_metrics = config
        self.train_metrics = config
        self.devices = config["devices"]

        self.config_as_dict = self.config.to_dict()
        self.wandb_logger = WandbLogger(
            log_model="all",
            project=config["experiment_name"],
            save_dir=config["experiment_log_dir"],
        )
        wandb.config = self.config_as_dict
        self.config.wandb_id = self.wandb_logger._wandb_init["id"]
        self.config.wandb_name = self.wandb_logger._wandb_init["name"]
        save_config(self.config.to_dict(), config["experiment_log_dir"] + "/config.yml")

        self.model = TransformerModule(config.model_config)

    @property
    def train_metrics(self):
        return self._train_metrics

    @train_metrics.setter
    def train_metrics(self, config):
        metrics = {metric: [] for metric in config["train_metrics"]}
        self._train_metrics = metrics

    @property
    def test_metrics(self):
        return self._test_metrics

    @test_metrics.setter
    def test_metrics(self, config):
        metrics = {metric: [] for metric in config["test_metrics"]}
        self._test_metrics = metrics

    def train(self, dataset: HuggingfaceDataset, **kwargs):
        callbacks = [
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                dirpath=f"{self.experiment_log_dir}/checkpoints/trainstep_checkpoints",
                filename="{epoch:06}-{step:09}",
                every_n_train_steps=self.config["batch_frequency"],
                save_last=True,
                verbose=True,
                save_weights_only=True,
            ),
            ModelCheckpoint(
                dirpath=f"{self.experiment_log_dir}/checkpoints",
                filename="{epoch:06}",
                verbose=True,
                save_last=True,
                save_on_train_epoch_end=True,
                save_weights_only=False,
            ),
        ]
        trainer_kwargs = self.config_as_dict["trainer_config"]
        trainer = Trainer(
            **trainer_kwargs, callbacks=callbacks, logger=self.wandb_logger
        )

        resume = kwargs.get("resume", None)
        train_dataset = dataset.train_dataloader

        try:
            resume_ckpt = None
            if resume is not None:
                resume_ckpt = os.path.join(resume, "checkpoints", "last.ckpt")
            trainer.fit(self.model, train_dataset, ckpt_path=resume_ckpt)
        finally:
            if trainer.global_rank == 0:
                final_ckpt = os.path.join(
                    self.experiment_log_dir, "checkpoints", "last.ckpt"
                )
                trainer.save_checkpoint(final_ckpt)

    def test(self, **kwargs):
        """
        To implement
        """
        pass

    def run_experiment(
        self, dataset: HuggingfaceDataset, resume: bool = None, **kwargs
    ):
        """
        Run experiment
        """
        results = self.train(dataset, resume=resume, **kwargs)
        return results
