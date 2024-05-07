import json
import os
import pickle
from typing import Optional

import omegaconf
import torch
import wandb
from tqdm import tqdm
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from transformers import AutoTokenizer, GenerationConfig, AutoConfig

from examples.bayes_llama3.llama3.modules.bayesllama import BayesLlamaForCausalLM
from examples.bayes_llama3.llama3.utils.load_utils import load_ensemble, save_config
from examples.bayes_llama3.llama3.utils.prompting import llama_chat_prompt, ONE_SHOT


class Experiment:
    def __init__(self, config: FrozenConfigDict):
        self.config = ConfigDict(config)  # thaw config
        save_config(
            self.config.to_dict(), config["experiment_log_dir"] + "/config.yaml"
        )
        self.experiment_log_dir = config.experiment_log_dir
        self.chat_model = config["chat_model"]
        self.n_tokens = config["n_tokens"]

        assert config["model_architecture"] == "llama"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_pretrained_model_name_or_path"]
        )
        self.generation_config = (
            GenerationConfig(config=config["generation_config"])
            if "generation_config" in config
            else None
        )
        model_dtype = torch.float16 if config["fp16"] else torch.float32
        if "model_config" in config:
            if "rope_scaling" in config["model_config"]:
                config["model_config"]["rope_scaling"] = dict(
                    config["model_config"].pop("rope_scaling")
                )
            model_config = AutoConfig.from_pretrained(
                config["pretrained_model_name_or_path"],
                **omegaconf.OmegaConf.to_container(
                    config["model_config"], resolve=True
                ),
            )

        bayes_config = config["bayes_config"]
        assert bayes_config['n_ensemble'] == len(config["checkpoint_paths"])
        parameters = load_ensemble(config["checkpoint_paths"])

        self.model = BayesLlamaForCausalLM.from_pretrained(
            config["pretrained_model_name_or_path"],
            torch_dtype=model_dtype,
            config=model_config if "model_config" in config else None,
            device_map=config["device_map"],
            bayes_config=config["bayes_config"],
        )
        self.model.load_bayesian_layers(parameters)

    def prepare_prompt(self, prompt, question, document):
        if self.chat_model:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt =  "\n".join([prompt, document, question])
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                prompt = llama_chat_prompt(
                        [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": "\n".join([document, question]),
                            },
                        ]
                    )                
                inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        else:
            question = question + "\nAnswer:"
            prompt = "\n".join([ONE_SHOT, prompt, document, question])
            inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

        print(prompt)
        return inputs
    
    def extract_questions(sample):
        questions = sample['questions']['nonDiagramQuestions']
        all_qs = []
        for key, value in questions.items():
            prompt = value['beingAsked']['processedText'] + '\n\n' #Question
            answers = value['answerChoices'] #Answers
            correct = value['correctAnswer']['processedText'] #Correct Answer
            for value in answers.values(): #Extract answer text
                answer = value['rawText']
                prompt += answer + '\n'
            all_qs.append((key, prompt, correct), )
        return all_qs
    
    def extract_context(sample):
        context =''
        topics = sample["topics"]
        for value in topics.values():
            if 'text' in value['content']:
                context += value['content']['text']
        return context

    @torch.no_grad()
    def generate(self, inputs, max_length=10, use_cache=True):
        seq_out = []
        for idx in range(max_length):
            outputs = self.model(**inputs, return_dict=False, use_cache=use_cache)

            if "attention_mask" in inputs:
                del inputs["attention_mask"]

            next_token = outputs[0][0][:, -1].argmax(-1).unsqueeze(-1)
            if use_cache:
                inputs["past_key_values"] = outputs[1]
                inputs["ensemble_past_key_values"] = outputs[2]
                inputs["input_ids"] = next_token
            else:
                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, -1)
        return self.tokenizer.batch_decode(seq_out, skip_special_tokens=True)


    def run_experiment(self, dataset_path: str):
        with open(dataset_path, "rb") as file:
            dataset = json.load(file)

        results = []
        for idx, sample in tqdm(enumerate(dataset)):
            questions = self.extract_questions(sample)
            q_ids = [q[0] for q in questions]
            question_text = [q[1] for q in questions]
            correct_answers = [q[2] for q in questions]

            context = self.extract_context(sample)
            prompt = "Answer the question based on the text below. You should answer the question by choosing the letter (a, b, c, or d) that corresponds with the correct answer.\n\n"
            inputs = [self.prepare_prompt(prompt=prompt, question=question, document=context) for question in question_text]

            ## These should all be done in batch
            inputs= inputs[0]

            out = self.generate(
                inputs.to(self.model.device),
                max_length=self.n_tokens
            )
            text = self.tokenizer.decode(out[0][-self.n_tokens :])
            eval = int(correct_answers[0] in text.lower())

            result = {"id": q_ids[0], "text": text, "eval": eval}
            results.append(sample | result)
            self.save_results(results)

        return results

    def save_results(
        self, results, metadata: str = None, checkpoint: Optional[int] = None
    ):  # To do: make more specific than pickle
        """
        Save results as pickle file
        """
        folder = "final" if checkpoint is None else f"checkpoints/{checkpoint}"
        experiment_dir = os.path.join(self.experiment_log_dir, folder)
        os.makedirs(experiment_dir, exist_ok=True)
        result_file = os.path.join(
            experiment_dir, f"results-{metadata}.pkl" if metadata else "results.pkl"
        )
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

    def run(self, dataset_path: str, **kwargs):
        """
        Run experiment
        """
        results = self.run_experiment(dataset_path, **kwargs)
        self.save_results(results)
        wandb.finish()

        return results
