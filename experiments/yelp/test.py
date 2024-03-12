import argparse
import pickle
import importlib
from tqdm import tqdm
import torch
from torch.distributions import Categorical
from optree import tree_map
import uqlib

from experiments.yelp.load import load_dataloaders, load_model
from experiments.yelp import utils

# Get config path and device from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--device", default="cpu", type=str)
args = parser.parse_args()

# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))

# Load data and model
# _, test_dataloader = load_dataloaders(
#     small=config.small_dataset, test_batch_size=config.test_batch_size
# )
_, test_dataloader = load_dataloaders(
    small=True, test_batch_size=config.test_batch_size
)
num_data = len(test_dataloader.dataset)
model, _ = load_model()
model_function = uqlib.model_to_function(model)
model.to(args.device)
print("Device: ", model.device)

# Load trained state
state = pickle.load(open(config.save_dir + "/state.pkl", "rb"))
del state.aux
state = tree_map(lambda x: x.to(args.device), state)


def test_metrics(logits, labels):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    expected_probs = probs.mean(dim=1)
    expected_logits = torch.log(expected_probs)
    loss = -Categorical(logits=expected_logits).log_prob(labels)
    accuracy = (expected_probs.argmax(dim=-1) == labels).float()
    expected_loss = -Categorical(logits=logits).log_prob(labels.unsqueeze(1)).mean(1)
    expected_accuracy = (probs.argmax(dim=-1) == labels.unsqueeze(1)).float().mean(1)

    expected_probs = torch.where(expected_probs < 1e-5, 1e-5, expected_probs)
    probs = torch.where(probs < 1e-5, 1e-5, probs)

    total_uncertainty = -(torch.log(expected_probs) * expected_probs).mean(1)
    aleatoric_uncertainty = -(torch.log(probs) * probs).mean(2).mean(1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return {
        "loss": loss,
        "accuracy": accuracy,
        "expected_loss": expected_loss,
        "expected_accuracy": expected_accuracy,
        "total_uncertainty": total_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "epistemic_uncertainty": epistemic_uncertainty,
    }


# Run through test data
num_batches = len(test_dataloader)
log_dict = {
    "loss": [],
    "accuracy": [],
    "expected_loss": [],
    "expected_accuracy": [],
    "total_uncertainty": [],
    "aleatoric_uncertainty": [],
    "epistemic_uncertainty": [],
}
log_dict_linearised = {k: [] for k in log_dict.keys()}
log_bar = tqdm(total=0, position=1, bar_format="{desc}")


for batch in tqdm(test_dataloader, position=0):
    with torch.no_grad():
        batch = tree_map(lambda x: x.to(args.device), batch)

        if hasattr(config.method, "sample"):
            sd_diag = config.to_sd_diag(state)
            param_samples = uqlib.diag_normal_sample(
                state.params, sd_diag, (config.n_test_samples,)
            )

            logits = torch.vmap(lambda params: model_function(params, **batch))(
                param_samples
            ).logits
            logits = logits.transpose(0, 1)

            lin_mean, lin_chol, _ = uqlib.linearized_forward_diag(
                lambda p, b: (model_function(p, **b).logits, torch.tensor([])),
                state.params,
                batch,
                sd_diag,
            )
            samps = torch.randn(
                lin_mean.shape[0],
                config.n_test_samples,
                lin_mean.shape[1],
                device=lin_mean.device,
            )
            lin_logits = lin_mean.unsqueeze(1) + samps @ lin_chol.transpose(-1, -2)

        else:
            logits = model_function(state.params, **batch).logits.unsqueeze(1)
            lin_logits = logits

        # Calculate metrics
        labels = batch["labels"]

        metrics = test_metrics(logits, labels)
        lin_metrics = test_metrics(lin_logits, labels)

        for k in log_dict.keys():
            log_dict[k] += metrics[k].tolist()
            log_dict_linearised[k] += lin_metrics[k].tolist()

        # Update metrics
        log_bar.set_description_str(
            f"Loss: {log_dict['loss'][-1]:.2f}, Accuracy: {log_dict['accuracy'][-1]:.2f}"
        )

utils.log_metrics(
    log_dict, config.save_dir, window=config.log_window, file_name="test", plot=False
)
utils.log_metrics(
    log_dict_linearised,
    config.save_dir,
    window=config.log_window,
    file_name="linearised_test",
    plot=False,
)
