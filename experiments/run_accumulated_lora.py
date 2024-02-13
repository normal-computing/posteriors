import argparse
import os
import glob
import datetime
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from datasets import load_dataset
from transformers import AutoTokenizer
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm
import wandb
import pickle

from experiments.utils import parse_devices, load_config, save_config, setup_log_dir
from experiments.laplace_lora import TransformerModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--base", default=None, type=str)
parser.add_argument("--devices", default=parse_devices, type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--log_frequency", default=10, type=int)
parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()


def evaluate(model, dataset, step):
    results = []
    avg_nlls = ()
    avg_ppls = ()
    for idx, sample in tqdm(enumerate(dataset)):
        input_ids = sample["input_ids"].unsqueeze(0)
        max_input_length = model.config.max_position_embeddings

        sample_nlls = []
        prev_end_loc = 0
        seq_len = input_ids.size(1)
        for begin_loc in range(0, seq_len, 512):
            end_loc = min(begin_loc + max_input_length, seq_len)
            subseq = input_ids[:, begin_loc:end_loc]
            targets = subseq.clone()
            trg_len = end_loc - prev_end_loc
            targets[:, :-trg_len] = -100

            with torch.no_grad():
                output = model.model(
                    input_ids=subseq.to(model.model.device),
                    labels=targets,
                )
                sample_nlls.append(output.loss)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        sample_nlls = torch.tensor(sample_nlls)
        sample_ppls = torch.exp(sample_nlls)

        sample_avg_nll = torch.mean(sample_nlls)
        sample_avg_ppl = torch.mean(sample_ppls)
        wandb.log({"sample_avg_nll": sample_avg_nll, "Task idx": step})
        wandb.log({"sample_avg_ppl": sample_avg_ppl, "Task idx": step})

        results += [
            {
                "idx": idx,
                "input_ids": sample["input_ids"],
                "nlls": sample_nlls,
                "ppls": sample_ppls,
                "avg_nll": sample_avg_nll,
                "avg_ppl": sample_avg_ppl,
            }
        ]
        avg_nlls += (sample_avg_nll,)
        avg_ppls += (sample_avg_ppl,)

    avg_nll = torch.mean(torch.tensor(avg_nlls))
    avg_ppl = torch.mean(torch.tensor(avg_ppls))

    wandb.log({"Avg NLL, Base Model": avg_nll, "Task idx": step})
    wandb.log({"Avg PPL, Base Model": avg_ppl, "Task idx": step})

    return results




if __name__ == "__main__":
    device_type = "cpu" if callable(args.devices) else "gpu"
    if args.resume is None:
        assert (
            args.base is not None
        ), "Configs not specified, specify at least resume or base"
        config = load_config(args.base)
    else:
        assert os.path.exists(
            args.resume
        ), "Provided path to resume training does not exist"
        config_paths = glob.glob(os.path.join(args.resume, "*.yaml"))
        assert len(config_paths) == 1, "Too many possible configs to resume from"
        config = load_config(config_paths[0])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment_name = config.get("experiment_name", None)

    experiment_log_dir = setup_log_dir(
        config.get("logs_dir", "logs"),
        timestamp,
        resume=args.resume,
        experiment_name=experiment_name,
    )

    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(args.seed)

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "accelerator": device_type,
        "log_every_n_steps": args.log_frequency,
    }
    
    model = TransformerModule(config.model_config)

    config = ConfigDict(config) #thaw
    logger = WandbLogger(
        log_model="all",
        project=config.get("experiment_name", ""),
        save_dir=config.get("logs_dir", "logs"),
    )
    config["wandb_name"] = logger.experiment.name
    config["wandb_id"] = logger.experiment.id

    config['epochs'] = args.epochs
    config['log_frequency'] = args.log_frequency
    config['seed'] = args.seed

    if args.resume is None:
        save_config(
            config.to_dict(), f"{experiment_log_dir}/{os.path.basename(args.base)}"
        )

    trainer = Trainer(**trainer_kwargs, logger=logger)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.pretrained_model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    # train[test]_dataloaders has size (num_tasks, num_samples_per_task*task_idx)
    with open(config.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    def split_doc(lst):
        start_index = int(len(lst) * 0.85)
        return lst[:start_index], lst[start_index:]

    for sample in dataset:
        sample['input_ids'] = tokenizer(sample['text'], padding="max_length", max_length=config.max_length, truncation=True,)['input_ids']

    num_samples_per_task = config.num_samples_per_task
    num_tasks = config.num_tasks

    samples_per_task = [dataset[(i * num_samples_per_task):((i + 1) * num_samples_per_task)] for i in range(num_tasks)]
    split_samples = [[split_doc(sample['input_ids']) for sample in samples] for samples in samples_per_task]
    train_datasets = [[sample[0] for sample in samples] for samples in split_samples]
    test_datasets = [[sample[1] for sample in samples] for samples in split_samples]

    train_datasets = [[x for xs in train_datasets[:i] for x in xs] for i in range(1, len(train_datasets)+1)]
    test_datasets = [[x for xs in test_datasets[:i] for x in xs] for i in range(1, len(test_datasets)+1)]

    train_dataloaders = [torch.utils.data.DataLoader(
        train,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    ) for train in train_datasets]

    test_dataloaders = [torch.utils.data.DataLoader(
        test,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    ) for test in test_datasets]


    # Models                         Eval 
    # Model1 = Tune on A(Model 0)       (On hold out from A)
    # Model2 = Tune on A, B(Model 0)    (On hold out from A, on hold out from B)
    # ...                            ....

    for step in range(num_tasks):
        try:
            resume_ckpt = None
            if args.resume is not None:
                resume_ckpt = os.path.join(args.resume, "checkpoints", "last.ckpt")
            trainer.fit(model, train_dataloaders[step], ckpt_path=resume_ckpt)
        finally:
            if trainer.global_rank == 0:
                final_ckpt = os.path.join(experiment_log_dir, "checkpoints", "last.ckpt")
                trainer.save_checkpoint(final_ckpt)

        LORA_WEIGHTS = experiment_log_dir + "/checkpoints/last.ckpt"
        model_tuned = TransformerModule.load_from_checkpoint(
            LORA_WEIGHTS, config=config["model_config"]
        ).to(args.devices[0])
        print("Weights loaded successfully!")
        
        eval_results = eval(model_tuned, test_dataloaders[step], step)

        result_file = os.path.join(experiment_log_dir, "evaluations","results-eval.pkl")
        with open(result_file, "wb") as f:
            pickle.dump(eval_results, f)
