import torch
import pickle

import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

oos_texts = pickle.load(open("oos_texts.pkl", "rb"))

# mod_name = "meta-llama/Llama-2-7b-hf"
mod_name = "meta-llama/Llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(mod_name)
tokenizer.pad_token = tokenizer.eos_token
hf_model = AutoModelForCausalLM.from_pretrained(mod_name, device_map="auto")


def token_ids_to_last_hidden_state(
    model,
    tokenizer,
    text,
) -> torch.LongTensor:
    input = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

    outputs = model(
        input["input_ids"],
        input["attention_mask"],
        return_dict=True,
        output_attentions=False,
        output_hidden_states=True,
    )

    last_hidden_state = outputs.hidden_states[-1].cpu()
    last_hidden_state_unpadded = [
        last_hidden_state[i, :input["attention_mask"][i].sum(), :]
        for i in range(last_hidden_state.shape[0])
    ]
    
    del input, outputs, last_hidden_state

    return last_hidden_state_unpadded


batchsize = 1

oos_hidden_states = []

with torch.inference_mode():
    for i in tqdm.trange(0, len(oos_texts), batchsize):
        end = min(i + batchsize, len(oos_texts))
        oos_hidden_states += token_ids_to_last_hidden_state(
            hf_model, tokenizer, oos_texts[i:end]
        )

pickle.dump(oos_hidden_states, open("oos_hidden_states.pkl", "wb"))

# oos_hidden_states = pickle.load(open("oos_hidden_states.pkl", "rb"))
