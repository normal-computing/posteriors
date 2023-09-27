import torch
import pickle

import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


data_path = "data/oos_train_texts.pkl"
# data_path = "data/oos_test_texts.pkl"
# data_path = "data/oos_oos_texts.pkl"

oos_texts = pickle.load(open(data_path, "rb"))

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


batchsize = 5

oos_hidden_states = []

with torch.inference_mode():
    for i in tqdm.trange(0, len(oos_texts), batchsize):
        end = min(i + batchsize, len(oos_texts))
        oos_hidden_states += token_ids_to_last_hidden_state(
            hf_model, tokenizer, oos_texts[i:end]
        )


save_path = data_path.replace(".pkl", "_hidden_states.pkl")
pickle.dump(oos_hidden_states, open(save_path, "wb"))

# oos_hidden_states = pickle.load(open(data_path.replace(save_path, "rb"))
