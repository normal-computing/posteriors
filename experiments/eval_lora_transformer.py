import torch
import wandb
from tqdm import tqdm
import pickle
from omegaconf import OmegaConf
import os
from ml_collections.config_dict import ConfigDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.laplace_lora import BayesTransformerModule
from experiments.utils.utils import load_config


def evaluate(model, tuned, dataset):
    results = []
    avg_nlls_base = ()
    avg_ppls_base = ()
    avg_nlls_tuned = ()
    avg_ppls_tuned = ()
    for idx, sample in tqdm(enumerate(dataset)):
        input_ids = sample["input_ids"].unsqueeze(0)
        max_input_length = model.config.max_position_embeddings

        sample_nlls_base = []
        sample_nlls_tuned = []
        prev_end_loc = 0
        seq_len = input_ids.size(1)
        for begin_loc in range(0, seq_len, 512):
            end_loc = min(begin_loc + max_input_length, seq_len)
            subseq = input_ids[:, begin_loc:end_loc]
            targets = subseq.clone()
            trg_len = end_loc - prev_end_loc
            targets[:, :-trg_len] = -100

            with torch.no_grad():
                output_base = model(
                    input_ids=subseq.to(model.device),
                    labels=targets,
                )
                sample_nlls_base.append(output_base.loss)

                output_tuned = tuned.model(
                    input_ids=subseq.to(tuned.model.device),
                    labels=targets,
                )
                sample_nlls_tuned.append(output_tuned.loss)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        sample_nlls_base = torch.tensor(sample_nlls_base)
        sample_ppls_base = torch.exp(sample_nlls_base)

        sample_avg_nll_base = torch.mean(sample_nlls_base)
        sample_avg_ppl_base = torch.mean(sample_ppls_base)
        wandb.log({"sample_avg_nll_base": sample_avg_nll_base})
        wandb.log({"sample_avg_ppl_base": sample_avg_ppl_base})

        sample_nlls_tuned = torch.tensor(sample_nlls_tuned)
        sample_ppls_tuned = torch.exp(sample_nlls_tuned)

        sample_avg_nll_tuned = torch.mean(sample_nlls_tuned)
        sample_avg_ppl_tuned = torch.mean(sample_ppls_tuned)
        wandb.log({"sample_avg_nll_tuned": sample_avg_nll_tuned})
        wandb.log({"sample_avg_ppl_tuned": sample_avg_ppl_tuned})

        results += [
            {
                "idx": idx,
                "input_ids": sample["input_ids"],
                "nlls_base": sample_nlls_base,
                "nlls_tuned": sample_nlls_tuned,
                "ppls_base": sample_ppls_base,
                "ppls_tuned": sample_ppls_tuned,
                "avg_nll_base": sample_avg_nll_base,
                "avg_ppl_base": sample_avg_ppl_base,
                "avg_nll_tuned": sample_avg_nll_tuned,
                "avg_ppl_tuned": sample_avg_ppl_tuned,
            }
        ]

        avg_nlls_base += (sample_avg_nll_base,)
        avg_ppls_base += (sample_avg_ppl_base,)

        avg_nlls_tuned += (sample_avg_nll_tuned,)
        avg_ppls_tuned += (sample_avg_ppl_tuned,)

    avg_nll_base = torch.mean(torch.tensor(avg_nlls_base))
    avg_ppl_base = torch.mean(torch.tensor(avg_ppls_base))

    avg_nll_tuned = torch.mean(torch.tensor(avg_nlls_tuned))
    avg_ppl_tuned = torch.mean(torch.tensor(avg_ppls_tuned))

    wandb.log({"Avg NLL, Base Model": avg_nll_base})
    wandb.log({"Avg PPL, Base Model": avg_ppl_base})

    wandb.log({"Avg NLL, Tuned Model": avg_nll_tuned})
    wandb.log({"Avg PPL, Tuned Model": avg_ppl_tuned})

    return results


DATETIME = ""
EXPERIMENT_LOG_DIR = f"./experiments/runs/laplace_lora/{DATETIME}_laplace_lora"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LORA_WEIGHTS = EXPERIMENT_LOG_DIR + "/checkpoints/last.ckpt"
    CONFIG = EXPERIMENT_LOG_DIR + "/laplace_lora.yaml"
    config = ConfigDict(load_config(CONFIG))

    model_tuned = BayesTransformerModule.load_from_checkpoint(
        LORA_WEIGHTS, config=config["model_config"]
    ).to(device)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_config.pretrained_model_name_or_path
    ).to(device)
    print("Weights loaded successfully!")

    wandb.init(
        project=config["experiment_name"],
        dir=config.get("logs_dir", "logs"),
    )
    config.wandb_id_eval = wandb.run.id
    config.wandb_name_eval = wandb.run.name

    OmegaConf.save(
        config=config.to_dict(),
        f=EXPERIMENT_LOG_DIR + f"/{config['experiment_name']}.yaml",
    )

    dataset = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_pretrained_model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=config.max_length,
            truncation=True,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns([config.inputs_key])
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    if config.small:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    results = evaluate(model, model_tuned, eval_dataset)

    result_file = os.path.join(EXPERIMENT_LOG_DIR, "results-eval.pkl")
    with open(result_file, "wb") as f:
        pickle.dump(results, f)
