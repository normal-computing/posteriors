from sghmc.modules.classifier import Classifier
import torch

from utils.utils import load_optimizer_param_to_model

ckpt_path = None

model = Classifier()
groups = torch.load(ckpt_path)["save_params"][9]
load_optimizer_param_to_model(model, groups)


print("Weights loaded successfully!")
