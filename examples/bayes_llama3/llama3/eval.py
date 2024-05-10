import json
import os
import pickle

import torch
from tqdm import tqdm
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy import special

from llama3.modules.bayesllama import BayesLlamaForCausalLM
from llama3.utils.load_utils import load_ensemble


PROMPT = "Answer the following multiple choice question very succintly.\n"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1


def compute_uncertainties(logits):
    # There is no notion of batch size in this function
    # Logits is a list of logits where each idx corresponds to the ith token output
    # The shape at index i is just (n ensemble, vocab size)
    prevent_zeros = lambda x: np.where(x < 1e-5, 1e-5, x)
    normalize_logits = lambda x: x / np.sum(x, axis=-1, keepdims=True)
    avg_ensemble = lambda x: np.mean(x, axis=0)
    apply_softmax = lambda x: special.softmax(x, axis=-1)

    calc_total_uncertainty = lambda x: -np.mean(np.log(x) * x)
    calc_aleatoric_uncertainty = lambda x: -np.mean(np.log(x) * x, axis=-1)

    probs = list(
        map(normalize_logits, map(prevent_zeros, map(apply_softmax, logits)))
    )  # List[(n ensemble, vocab size)]
    expected_probs = list(map(avg_ensemble, probs))  # List[(vocab size, )]

    total_uncertainty = list(map(calc_total_uncertainty, expected_probs))
    aleatoric_uncertainty_per_ensemble = list(map(calc_aleatoric_uncertainty, probs))
    aleatoric_uncertainty = [np.mean(uc) for uc in aleatoric_uncertainty_per_ensemble]
    epistemic_uncertainty = [
        t - a for t, a in zip(total_uncertainty, aleatoric_uncertainty)
    ]

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


class Experiment:
    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)  # thaw config
        self.experiment_log_dir = config.experiment_log_dir
        self.n_tokens = config["n_tokens"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_pretrained_model_name_or_path"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.eval_pretrained = config.get("eval_pretrained_model", False)
        if self.eval_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"]
            )
        else:
            assert os.path.isdir(
                config["checkpoints_folder"]
            ), "Provided checkpoints is not a path to a folder"
            checkpoints = [
                os.path.join(config["checkpoints_folder"], path)
                for path in os.listdir(config["checkpoints_folder"])
                if path.endswith(".ckpt")
            ]
            parameters = load_ensemble(checkpoints)

            self.model = BayesLlamaForCausalLM.from_pretrained(
                config["pretrained_model_name_or_path"],
                bayes_config={"n_ensemble": len(checkpoints)},
            )
            self.model.load_bayesian_layers(parameters)

        self.model.to(DEVICE)

    def prepare_prompt(self, prompt, questions):
        prompt = ["\n".join([prompt, q + "\n", "Answer:"]) for q in questions]
        return self.tokenizer(prompt, return_tensors="pt", padding=True)

    def extract_questions(self, sample):
        all_qs = []
        for key, val in sample["questions"]["nonDiagramQuestions"].items():
            correct_answer = val["correctAnswer"]["processedText"]
            if correct_answer not in val["answerChoices"]:
                # Adding catch
                continue
            correct_answer = val["answerChoices"][correct_answer]["rawText"]
            question = val["beingAsked"]["processedText"]
            for _, mc in val["answerChoices"].items():
                question += "\n" + mc["idStructural"] + " " + mc["processedText"]
            all_qs.append((key, question, correct_answer))
        return all_qs

    @torch.no_grad()
    def generate(self, inputs, max_length=20, use_cache=True):
        seq_out, ensemble_logits = [], [[] for _ in range(inputs["input_ids"].size(0))]
        for _ in range(max_length):
            outputs = self.model(**inputs, return_dict=False, use_cache=use_cache)

            if "attention_mask" in inputs:
                del inputs["attention_mask"]

            next_token = outputs[0][0][:, -1].argmax(-1).unsqueeze(-1)
            logits = (
                outputs[0][1].swapaxes(0, 1)[:, :, -1].cpu().numpy()
            )  # swap to make batch dim first
            for idx, logit in enumerate(logits):
                ensemble_logits[idx].append(logit)

            if use_cache:
                inputs["past_key_values"] = outputs[1]
                inputs["ensemble_past_key_values"] = outputs[2]
                inputs["input_ids"] = next_token
            else:
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], next_token], dim=1
                )
            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, -1)
        text_outputs = self.tokenizer.batch_decode(seq_out, skip_special_tokens=True)
        return text_outputs, ensemble_logits

    @torch.no_grad()
    def generate_base(self, inputs, max_length=20, use_cache=True):
        seq_out, logits = [], [[] for _ in range(inputs["input_ids"].size(0))]
        for _ in range(max_length):
            outputs = self.model(**inputs, return_dict=True, use_cache=use_cache)

            if "attention_mask" in inputs:
                del inputs["attention_mask"]

            next_token = outputs.logits[:, -1].argmax(-1).unsqueeze(-1)
            out_logits = outputs.logits[:, -1].cpu().numpy()
            for idx, logit in enumerate(out_logits):
                logits[idx].append([logit])

            if use_cache:
                inputs["past_key_values"] = outputs.past_key_values
                inputs["input_ids"] = next_token
            else:
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], next_token], dim=1
                )
            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, -1)
        text_outputs = self.tokenizer.batch_decode(seq_out, skip_special_tokens=True)
        return text_outputs, logits

    def run_experiment(self, dataset_path: str, split: str):
        with open(dataset_path, "rb") as file:
            dataset = json.load(file)

        results = {}
        for sample_idx, sample in tqdm(list(enumerate(dataset))):
            sample = self.extract_questions(sample)

            for idx in range(0, len(sample), BATCH_SIZE):
                batch_sample = sample[idx : idx + BATCH_SIZE]

                q_ids = [q[0] for q in batch_sample]
                questions = [q[1] for q in batch_sample]
                correct_answers = [q[2] for q in batch_sample]

                inputs = self.prepare_prompt(PROMPT, questions)

                if self.eval_pretrained:
                    answers, logits = self.generate_base(
                        inputs.to(self.model.device), max_length=self.n_tokens
                    )
                else:
                    answers, logits = self.generate(
                        inputs.to(self.model.device), max_length=self.n_tokens
                    )

                for qid, answer, ca, q, logit in zip(
                    q_ids, answers, correct_answers, questions, logits
                ):
                    total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = (
                        compute_uncertainties(logit)
                    )
                    results[qid] = {
                        "question": q,
                        "response": answer,
                        "expected_response": ca,
                        # "logits": logit,
                        "total_uncertainty": total_uncertainty,
                        "aleatoric_uncertainty": aleatoric_uncertainty,
                        "epistemic_uncertainty": epistemic_uncertainty,
                    }

            if sample_idx % 10 == 0:
                self.save_results(results, split)

        return results

    def save_results(self, results, split):  # To do: make more specific than pickle
        """
        Save results as pickle file
        """
        result_file = os.path.join(self.experiment_log_dir, f"results_{split}.pkl")
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

    def run(self, dataset_path: str, **kwargs):
        """
        Run experiment
        """
        split = os.path.basename(dataset_path).split("_")[-1].split(".")[0]
        results = self.run_experiment(dataset_path, split, **kwargs)
        self.save_results(results, split)

        return results
