import os

import torch
import torch.nn.functional as F
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from llama3.data.statements import scientific_statements, scientific_statements_samoan
from llama3.modules.bayesllama import BayesLlamaForCausalLM
from llama3.utils.load_utils import load_ensemble


PROMPT = "Complete the following statement:\n"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
eps = 1e-5


def logits_to_uncertainties(eprobs):
    probs = eprobs.mean(0)
    total_uncertainty = -torch.sum(probs * torch.log(probs), -1)
    aleatoric_uncertainty = -(eprobs * torch.log(eprobs)).sum(-1).mean(0)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    return total_uncertainty, epistemic_uncertainty


class EvaluationEngine:
    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)
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

    @torch.no_grad()
    def generate(self, inputs, gt_token, max_length=20, use_cache=True, is_base=False):
        accuracy = 0.0
        prediction_loss = []
        epistemic_uncertainties = [[] for _ in range(inputs["input_ids"].size(0))]
        total_uncertainties = [[] for _ in range(inputs["input_ids"].size(0))]

        predictions = []
        for token_idx in range(max_length):
            outputs = self.model(**inputs, return_dict=False, use_cache=use_cache)

            if "attention_mask" in inputs:
                del inputs["attention_mask"]

            if is_base:
                # outputs is of shape (batch size, seq length, vocab size)
                elogits = outputs[0][:, -1].unsqueeze(0)
            else:
                # outputs is of shape (ensemble size, batch size, seq length, vocab size)
                elogits = outputs[0][:, :, -1]

            eprobs = torch.softmax(elogits, dim=-1)
            probs = eprobs.mean(0)

            # (batch size, vocab_size)
            pred_logits = torch.log(probs)
            next_token = probs.argmax(-1).unsqueeze(-1)

            if token_idx == 0:
                accuracy += torch.where(next_token.squeeze(-1).cpu() == gt_token)[
                    0
                ].numel()
                loss = F.cross_entropy(pred_logits.cpu(), gt_token)
                prediction_loss.append(loss.item())

            total_unc, epi_unc = logits_to_uncertainties(eprobs.cpu())
            for idx, (tunc, eunc) in enumerate(zip(total_unc, epi_unc)):
                total_uncertainties[idx].append(tunc.item())
                epistemic_uncertainties[idx].append(eunc.item())

            if use_cache:
                inputs["past_key_values"] = outputs[1]
                if not is_base:
                    inputs["ensemble_past_key_values"] = outputs[2]
                inputs["input_ids"] = next_token
            else:
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], next_token], dim=1
                )
            predictions.append(next_token)

        predictions = torch.cat(predictions, -1)
        text_outputs = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        return (
            text_outputs,
            total_uncertainties,
            epistemic_uncertainties,
            prediction_loss,
            accuracy,
        )

    def run_eval(self, statements, batch_size=3, n_tokens=5):
        model_outputs = []
        accuracies = 0.0
        uncertainties = {"epistemic": [], "total": []}
        loss_metrics = []
        for i in range(0, len(statements), batch_size):
            last_words = [
                " " + s.split(" ")[-1] for s in statements[i : i + batch_size]
            ]
            batch_statements = [
                " ".join(s.split(" ")[:-1]) for s in statements[i : i + batch_size]
            ]

            self.tokenizer.padding_side = "right"
            last_token = self.tokenizer(last_words, padding=True, return_tensors="pt")[
                "input_ids"
            ][:, 1]
            self.tokenizer.padding_side = "left"

            inputs = self.tokenizer(
                [PROMPT + s + " " for s in batch_statements],
                padding=True,
                return_tensors="pt",
            )
            (
                text_outputs,
                total_uncertainty,
                epistemic_uncertainty,
                batch_loss,
                batch_accuracy,
            ) = self.generate(
                inputs.to("cuda"),
                last_token,
                max_length=n_tokens,
                use_cache=True,
                is_base=self.eval_pretrained,
            )
            loss_metrics.extend(batch_loss)
            accuracies += batch_accuracy

            for text_output, total_unc, epi_unc in zip(
                text_outputs, total_uncertainty, epistemic_uncertainty
            ):
                model_outputs.append(text_output)
                uncertainties["total"].append(total_unc)
                uncertainties["epistemic"].append(epi_unc)

        return (
            model_outputs,
            uncertainties,
            np.average(loss_metrics),
            accuracies / len(statements),
        )

    def run(self, n_tokens):
        statements = [
            (scientific_statements, "en"),
            (scientific_statements_samoan, "sa"),
        ]

        results = {}
        for statement_list, lang in statements:
            outputs, uncertainties, loss, acc = self.run_eval(
                statement_list, n_tokens=n_tokens
            )
            results[lang] = {
                "outputs": outputs,
                "uncertainties": uncertainties,
                "loss": loss,
                "acc": acc,
            }

        return results
