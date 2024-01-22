import torch.nn as nn

num_classes = 151
d_model = 5120
num_layers = 2


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, inputs):
        output = self.model(inputs)
        return output


if __name__ == "__main__":
    import numpy as np

    model = Classifier()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of parameters = {n_params/1e6} million")
