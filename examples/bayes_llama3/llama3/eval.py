import json
import os
import pickle

import torch
from tqdm import tqdm
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from transformers import AutoTokenizer
import numpy as np
from scipy import special

from llama3.modules.bayesllama import BayesLlamaForCausalLM
from llama3.utils.load_utils import load_ensemble
from llama3.utils.prompting import llama_chat_prompt


PROMPT = "Answer the following multiple choice question. You should answer the question by choosing the letter (a, b, c, or d) that corresponds with the correct answer.\n\n"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 5


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
        self.chat_model = config["chat_model"]
        self.n_tokens = config["n_tokens"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_pretrained_model_name_or_path"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        assert os.path.isdir(config["checkpoints_folder"]), "Provided "
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
        if self.chat_model:
            prompts = [
                llama_chat_prompt(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": q},
                    ]
                )
                for q in questions
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        else:
            prompt = ["\n".join([prompt, q, "Answer:"]) for q in questions]
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        return inputs

    def extract_questions(self, sample):
        all_qs = []
        for key, val in sample["questions"]["nonDiagramQuestions"].items():
            correct_answer = val["correctAnswer"]["processedText"]
            correct_answer = val["answerChoices"][correct_answer]["rawText"]
            question = val["beingAsked"]["processedText"]
            for _, mc in val["answerChoices"].items():
                question += "\n" + mc["idStructural"] + " " + mc["processedText"]
            all_qs.append((key, question, correct_answer))
        return all_qs

    @torch.no_grad()
    def generate(self, inputs, max_length=10, use_cache=True):
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

    def run_experiment(self, dataset_path: str):
        with open(dataset_path, "rb") as file:
            dataset = json.load(file)

        results = {}
        for _, sample in tqdm(list(enumerate(dataset))):
            sample = self.extract_questions(sample)

            for idx in range(0, len(sample), BATCH_SIZE):
                batch_sample = sample[idx : idx + BATCH_SIZE]

                q_ids = [q[0] for q in batch_sample]
                questions = [q[1] for q in batch_sample]
                correct_answers = [q[2] for q in batch_sample]

                # context = self.extract_context(sample)
                inputs = self.prepare_prompt(PROMPT, questions)
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

        return results

    def evaluate_response():
        pass

    def save_results(self, results):  # To do: make more specific than pickle
        """
        Save results as pickle file
        """
        result_file = os.path.join(self.experiment_log_dir, "results.pkl")
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

    def run(self, dataset_path: str, **kwargs):
        """
        Run experiment
        """
        results = self.run_experiment(dataset_path, **kwargs)
        self.save_results(results)

        return results