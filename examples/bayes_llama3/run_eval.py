import argparse
import pickle

from llama3.eval import EvaluationEngine
from llama3.utils.load_utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

config = load_config(args.base)
experiment = EvaluationEngine(config["experiment_config"])
results = experiment.run(n_tokens=1)

with open(args.output, "wb") as f:
    pickle.dump(results, f)
