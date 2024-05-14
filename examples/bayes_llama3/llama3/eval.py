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
eps = 1e-5


def logits_to_uncertainties(eprobs):
    probs = eprobs.mean(0)
    total_uncertainty = -torch.sum(probs * torch.log(probs), -1)
    aleatoric_uncertainty = -(eprobs * torch.log(eprobs)).sum(-1).mean(0)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    return total_uncertainty, epistemic_uncertainty


def extract_tqa_questions(sample):
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


def extract_head_questions(sample):
    all_qs = []
    for q in sample["data"]:
        question = q["qtext"]
        for mc in q["answers"]:
            question += "\n" + f"{str(mc['aid'])}." + " " + mc["atext"]

        for answer in q["answers"]:
            if str(answer["aid"]) == str(q["ra"]):
                atext = answer["atext"]
        correct_answer = f"{str(q['ra'])}." + " " + atext
        all_qs.append((q["qid"], question, correct_answer))
    return all_qs


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

    @torch.no_grad()
    def generate(self, inputs, max_length=20, use_cache=True):
        seq_out = []
        epistemic_uncertainties = [[] for _ in range(inputs["input_ids"].size(0))]
        total_uncertainties = [[] for _ in range(inputs["input_ids"].size(0))]
        for _ in range(max_length):
            outputs = self.model(**inputs, return_dict=False, use_cache=use_cache)

            if "attention_mask" in inputs:
                del inputs["attention_mask"]

            elogits = outputs[0][:, :, -1]
            eprobs = torch.softmax(elogits, dim=-1)
            probs = eprobs.mean(0)
            next_token = probs.argmax(-1).unsqueeze(-1)

            total_unc, epi_unc = logits_to_uncertainties(eprobs.cpu())
            for idx, (tunc, eunc) in enumerate(zip(total_unc, epi_unc)):
                total_uncertainties[idx].append(tunc.item())
                epistemic_uncertainties[idx].append(eunc.item())

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
        return text_outputs, total_uncertainties, epistemic_uncertainties

    @torch.no_grad()
    def generate_base(self, inputs, max_length=20, use_cache=True):
        seq_out = []
        epistemic_uncertainties = [[] for _ in range(inputs["input_ids"].size(0))]
        total_uncertainties = [[] for _ in range(inputs["input_ids"].size(0))]
        for _ in range(max_length):
            outputs = self.model(**inputs, return_dict=True, use_cache=use_cache)

            if "attention_mask" in inputs:
                del inputs["attention_mask"]

            logits = outputs.logits[:, -1]
            probs = torch.softmax(logits, dim=-1)
            next_token = probs.argmax(-1).unsqueeze(-1)

            total_unc, epi_unc = logits_to_uncertainties(probs.unsqueeze(0).cpu())
            for idx, (tunc, eunc) in enumerate(zip(total_unc, epi_unc)):
                total_uncertainties[idx].append(tunc.item())
                epistemic_uncertainties[idx].append(eunc.item())

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
        return text_outputs, total_uncertainties, epistemic_uncertainties

    def run_experiment(self, dataset_name: str, dataset_path: str, split: str):
        with open(dataset_path, "rb") as file:
            dataset = json.load(file)
        if dataset_name == "head_qa":
            dataset = dataset["exams"].values()

        results = {}
        for sample_idx, sample in tqdm(list(enumerate(dataset))):
            if dataset_name == "head_qa":
                sample = extract_head_questions(sample)
            else:
                sample = extract_tqa_questions(sample)

            for idx in range(0, len(sample), BATCH_SIZE):
                batch_sample = sample[idx : idx + BATCH_SIZE]

                q_ids = [q[0] for q in batch_sample]
                questions = [q[1] for q in batch_sample]
                correct_answers = [q[2] for q in batch_sample]

                inputs = self.prepare_prompt(PROMPT, questions)

                if self.eval_pretrained:
                    answers, total_uncertainties, epistemic_uncertainties = (
                        self.generate_base(
                            inputs.to(self.model.device), max_length=self.n_tokens
                        )
                    )
                else:
                    answers, total_uncertainties, epistemic_uncertainties = (
                        self.generate(
                            inputs.to(self.model.device), max_length=self.n_tokens
                        )
                    )

                for qid, answer, ca, q, tunc, eunc in zip(
                    q_ids,
                    answers,
                    correct_answers,
                    questions,
                    total_uncertainties,
                    epistemic_uncertainties,
                ):
                    results[qid] = {
                        "question": q,
                        "response": answer,
                        "expected_response": ca,
                        "total_uncertainty": tunc,
                        "aleatoric_uncertainty": eunc,
                    }

            if sample_idx % 10 == 0:
                self.save_results(dataset_name, results, split)

        return results

    def save_results(self, dataset, results, split):
        """
        Save results as pickle file
        """
        result_file = os.path.join(
            self.experiment_log_dir, f"{dataset}_results_{split}.pkl"
        )
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

    def run(self, dataset_name: str, dataset_path: str, **kwargs):
        """
        Run experiment
        """
        splits = os.path.basename(dataset_path).split(".")[0].split("_")
        split = ""
        if "train" in splits:
            split = "train"
        elif "test" in splits:
            split = "test"
        results = self.run_experiment(dataset_name, dataset_path, split, **kwargs)
        self.save_results(dataset_name, results, split)

        return results
