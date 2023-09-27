import torch.nn as nn

num_classes = 150
d_model = 5120
num_layers = 2


# transformer_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=1, dropout=0)
# transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)


class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    import numpy as np

    model = MiniTransformer()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of parameters = {n_params/1e6} million")
