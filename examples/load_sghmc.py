from sghmc.modules.classifier import Classifier
from uqlib import load_optimizer_param_to_model
import torch

ckpt_path = (
    "/home/paperspace/Projects/bayes-lms/examples/lightning_logs/checkpoints/last.ckpt"
)

model = Classifier()
groups = torch.load(ckpt_path)["save_params"][9]
load_optimizer_param_to_model(model, groups)


print("Weights loaded successfully!")
