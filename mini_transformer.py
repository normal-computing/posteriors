import torch
import torch.nn as nn
import pickle
import numpy as np

# oos_texts = pickle.load(open("oos_texts.pkl", "rb"))
# oos_hidden_states = pickle.load(open("oos_hidden_states.pkl", "rb"))


num_classes = 150

d_model = 5120
num_layers = 2


model = nn.Linear(d_model, num_classes)


# transformer_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=1, dropout=0)
# transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
n_params = sum([np.prod(p.size()) for p in model_parameters])

print(f"Number of parameters = {n_params/1e6} million")






