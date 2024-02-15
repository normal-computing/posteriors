import os
from transformers import AutoTokenizer
from experiments.utils import load_config
from experiments.data.load_pg19 import load_pg19_dataloaders

config = load_config("experiments/utils/configs/lora_sam.yaml")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(
    config.model_config.pretrained_model_name_or_path
)

tokenizer.pad_token = tokenizer.eos_token

train_dataloaders, test_dataloaders = load_pg19_dataloaders(config, tokenizer)

batch = next(iter(train_dataloaders[0]))
