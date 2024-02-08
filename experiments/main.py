"""Script for running experiments."""

import datetime
import glob
import os
from absl import app, flags
from experiments.utils.utils import (
    load_config,
    save_config,
    setup_log_dir,
)
from experiments.laplace_lora import LoRAExperiment, HuggingfaceDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("base", None, "Path to base config.")
flags.DEFINE_string("resume", None, "Path to resume training.")
flags.DEFINE_string("devices", None, "Devices to use.")
flags.DEFINE_boolean("verbose", False, "Whether to print non-flag arguments.")


def main(argv):
    """
    Main function for running experiments.
    """
    if FLAGS.verbose:
        print("non-flag arguments:", argv)

    if FLAGS.resume is None:
        assert (
            FLAGS.base is not None
        ), "Configs not specified, specify at least resume or base"
        config = load_config(FLAGS.base)
    else:
        assert os.path.exists(
            FLAGS.resume
        ), "Provided path to resume training does not exist"
        config_paths = glob.glob(os.path.join(FLAGS.resume, "*.yaml"))
        assert len(config_paths) == 1, "Too many possible configs to resume from"
        config = load_config(config_paths[0])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment_name = config.get("experiment_name", None)

    experiment_log_dir = setup_log_dir(
        config.get("logs_dir", "logs"),
        timestamp,
        resume=FLAGS.resume,
        experiment_name=experiment_name,
    )
    if FLAGS.devices is not None:
        devices_list = FLAGS.devices.split(",")
        config["experiment_config"]["devices"] = devices_list

    if FLAGS.resume is None:
        config["experiment_config"]["experiment_log_dir"] = experiment_log_dir
        save_config(
            config.to_dict(), f"{experiment_log_dir}/{os.path.basename(FLAGS.base)}"
        )

    experiment = LoRAExperiment(
        config["experiment_config"]
    )  ## This will CHANGE per experiment
    dataset = HuggingfaceDataset(
        config["dataset_config"]
    )  ## This will CHANGE per experiment

    experiment.run(dataset=dataset, resume=FLAGS.resume)


if __name__ == "__main__":
    app.run(main)
