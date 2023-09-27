from sghmc.dataloader import ClincOOSDataLoader, ClincOOSDataset
from sghmc.modules.transformer import MiniTransformer
from sghmc.modules.optimizer import init, step, SGHMC


from torch.optim import SGD
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch


stepsize = 1e-2
alpha = 0.0
beta = 0.0
epochs = 100


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

data = DataLoader(
    ClincOOSDataset("data", "train"), batch_size=5000, shuffle=True, num_workers=8
)
model = MiniTransformer().to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=stepsize)
# optimizer = SGHMC(model.parameters(), lr=stepsize)


def train():
    init_params = [p.to(device) for p in model.parameters()]
    optimizer_state = init(init_params, alpha=alpha, beta=beta)
    all_params = [optimizer_state.params]

    for _ in range(epochs):
        for batch in data:
            # optimizer.zero_grad()

            x, y = batch
            outputs = model(x.to(device))
            loss = F.cross_entropy(outputs, y.squeeze(-1).to(device))

            loss.backward()
            # optimizer.step()

            gradients = [p.grad for p in model.parameters()]
            optimizer_state = step(optimizer_state, gradients, stepsize)
            all_params = all_params + [optimizer_state.params]

            for i, p in enumerate(model.parameters()):
                if i == 0:
                    print(p[:10])
                p.data = optimizer_state.params[i]
                p.grad.zero_()

            print(loss)


if __name__ == "__main__":
    train()


# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
# from pytorch_lightning.loggers import CSVLogger


# # log_dir_name = "logs/train-1"


# # logger = CSVLogger(name="csvlogger", save_dir=log_dir_name, version="")
# # callbacks = [
# #     TQDMProgressBar(refresh_rate=1),
# #     ModelCheckpoint(
# #         dirpath=f"{log_dir_name}/checkpoints/trainstep_checkpoints",
# #         filename="{epoch:06}-{step:09}",
# #         every_n_train_steps=10,
# #         save_last=True,
# #         verbose=True,
# #         save_weights_only=True,
# #     ),
# #     ModelCheckpoint(
# #         dirpath=f"{log_dir_name}/checkpoints",
# #         filename="{epoch:06}",
# #         verbose=True,
# #         save_last=True,
# #         save_on_train_epoch_end=True,
# #         save_weights_only=False,
# #     ),
# # ]


# trainer_kwargs = {
#     "max_epochs": 100,
#     "accelerator": "gpu",
#     "devices": [3],
#     "strategy": "ddp_find_unused_parameters_false",
#     "num_sanity_val_steps": 0,
# }

# trainer = pl.Trainer(**trainer_kwargs)
# trainer.fit(model, data)
