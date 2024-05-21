import argparse
import pickle
import importlib
from tqdm import tqdm
import torch
from torch.distributions import Categorical
from optree import tree_map
import os

from examples.imdb.model import CNNLSTM
from examples.imdb.data import load_imdb_dataset
from examples.imdb.utils import log_metrics

# Get config path and device from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--temperature", default=None, type=float)
args = parser.parse_args()


torch.manual_seed(args.seed)

# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))
config_dir = os.path.dirname(args.config)
save_dir = (
    config_dir
    if args.temperature is None
    else config_dir + f"_temp{str(args.temperature).replace('.', '-')}"
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load data
_, test_dataloader = load_imdb_dataset()

# Load model
model = CNNLSTM(num_classes=2)
model.to(args.device)

# Load state
state = pickle.load(open(config_dir + "/state.pkl", "rb"))
state = tree_map(lambda x: x.to(args.device), state)

# Dictionary containing forward functions
forward_dict = config.forward_dict


def test_metrics(logits, labels):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.where(probs < 1e-5, 1e-5, probs)
    probs /= probs.sum(dim=-1, keepdim=True)
    expected_probs = probs.mean(dim=1)

    logits = torch.log(probs)
    expected_logits = torch.log(expected_probs)

    loss = -Categorical(logits=expected_logits).log_prob(labels)
    accuracy = (expected_probs.argmax(dim=-1) == labels).float()

    total_uncertainty = -(torch.log(expected_probs) * expected_probs).mean(1)
    aleatoric_uncertainty = -(torch.log(probs) * probs).mean(2).mean(1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return {
        "loss": loss,
        "accuracy": accuracy,
        "total_uncertainty": total_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "epistemic_uncertainty": epistemic_uncertainty,
    }


# Run through test data
num_batches = len(test_dataloader)
metrics = [
    "loss",
    "accuracy",
    "total_uncertainty",
    "aleatoric_uncertainty",
    "epistemic_uncertainty",
]

log_dict_forward = {k: {m: [] for m in metrics} for k in forward_dict.keys()}

for batch in tqdm(test_dataloader):
    with torch.no_grad():
        batch = tree_map(lambda x: x.to(args.device), batch)
        labels = batch[1]

        forward_logits = {
            k: v(model, state, batch)
            if args.temperature is None
            else v(model, state, batch, args.temperature)
            for k, v in forward_dict.items()
        }

        forward_metrics = {
            k: test_metrics(logits, labels) for k, logits in forward_logits.items()
        }

        for forward_k in log_dict_forward.keys():
            for metric_k in metrics:
                log_dict_forward[forward_k][metric_k] += forward_metrics[forward_k][
                    metric_k
                ].tolist()


for forward_k, metric_dict in log_dict_forward.items():
    log_metrics(
        metric_dict,
        save_dir,
        file_name="test_" + forward_k,
        plot=False,
    )
