import argparse
import pickle
import importlib
from tqdm import tqdm
import torch
from torch.distributions import Categorical
from optree import tree_map
import uqlib

from experiments.yelp.load import load_dataloaders, load_model, load_spanish_dataloader
from experiments.yelp import utils

# Get config path and device from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--spanish", default=False, type=bool)
args = parser.parse_args()

# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))

# Load data
if args.spanish:
    test_dataloader = load_spanish_dataloader(
        small=True, batch_size=config.test_batch_size
    )
else:
    _, test_dataloader = load_dataloaders(
        small=True, test_batch_size=config.test_batch_size
    )


# Load model
num_data = len(test_dataloader.dataset)
if config.last_layer:
    model, _, _ = load_model(
        params_dir=config.params_dir,
        device=args.device,
    )
else:
    model, _, _ = load_model(
        params_dir=config.save_dir + "/state.pkl", device=args.device
    )
print("Device: ", model.device)


# Function that maps batch to pre-classifier layer
def penultimate_layer_func(batch):
    # From https://github.com/huggingface/transformers/blob/092f1fdaa4224fdd88c616dc9678e6fcb37bfffd/src/transformers/models/bert/modeling_bert.py#L1564
    bert_outputs = model.bert(**{k: v for k, v in batch.items() if k != "labels"})
    return bert_outputs[1]


# Function that maps last layer params and output from penultimate_layer to logits
last_layer_func = uqlib.model_to_function(model.classifier)


def sub_params_to_classifier_params(sub_params):
    return {
        "weight": sub_params["classifier.weight"],
        "bias": sub_params["classifier.bias"],
    }


# Load trained state
if hasattr(config, "combined_dir"):
    state = pickle.load(open(config.combined_dir + "/state.pkl", "rb"))

    def get_params():
        return state.params

elif hasattr(config, "to_sd_diag"):
    state = pickle.load(open(config.save_dir + "/state.pkl", "rb"))

    sd_diag = sub_params_to_classifier_params(config.to_sd_diag(state))
    sd_diag = tree_map(lambda x: x.to(args.device), sd_diag)

    def get_params():
        return uqlib.diag_normal_sample(state.params, sd_diag, (config.n_test_samples,))

else:
    state = pickle.load(open(config.save_dir + "/state.pkl", "rb"))

    def get_params():
        return tree_map(lambda x: x.unsqueeze(0), state.params)


state = tree_map(lambda x: x.to(args.device), state)
state.params = sub_params_to_classifier_params(state.params)


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
log_dict = {
    "loss": [],
    "accuracy": [],
    "total_uncertainty": [],
    "aleatoric_uncertainty": [],
    "epistemic_uncertainty": [],
}
if hasattr(config, "to_sd_diag"):
    log_dict_linearised = {k: [] for k in log_dict.keys()}

log_bar = tqdm(total=0, position=1, bar_format="{desc}")
for batch in tqdm(test_dataloader, position=0):
    with torch.no_grad():
        batch = tree_map(lambda x: x.to(args.device), batch)
        labels = batch["labels"]

        penultimate_layer = penultimate_layer_func(batch)

        ll_params = get_params()

        logits = torch.vmap(last_layer_func, in_dims=(0, None))(
            ll_params, penultimate_layer
        ).transpose(0, 1)

        metrics = test_metrics(logits, labels)

        for k in log_dict.keys():
            log_dict[k] += metrics[k].tolist()

        if hasattr(config, "to_sd_diag"):
            lin_mean, lin_chol, _ = uqlib.linearized_forward_diag(
                lambda p, pl: (last_layer_func(p, pl), torch.tensor([])),
                state.params,
                penultimate_layer,
                sd_diag,
            )
            samps = torch.randn(
                lin_mean.shape[0],
                config.n_linearised_test_samples,
                lin_mean.shape[1],
                device=lin_mean.device,
            )
            lin_logits = lin_mean.unsqueeze(1) + samps @ lin_chol.transpose(-1, -2)

            lin_metrics = test_metrics(lin_logits, labels)
            for k in log_dict_linearised.keys():
                log_dict_linearised[k] += lin_metrics[k].tolist()

        # Update metrics
        log_bar.set_description_str(f"Loss: {log_dict['loss'][-1]:.2f},")

utils.log_metrics(
    log_dict,
    config.test_save_dir,
    window=config.log_window,
    file_name="test_spanish" if args.spanish else "test",
    plot=False,
)

if hasattr(config, "to_sd_diag"):
    utils.log_metrics(
        log_dict_linearised,
        config.test_save_dir,
        window=config.log_window,
        file_name="linearised_test_spanish" if args.spanish else "linearised_test",
        plot=False,
    )
