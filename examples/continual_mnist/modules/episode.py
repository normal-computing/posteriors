import torch
from torch import func


def train_for_episode(params, optimizer, data, label, device, epochs=5, batch_size=100):
    state = optimizer.init(params)

    for _ in range(epochs):
        for i in range(0, len(data), batch_size):
            x = torch.stack([torch.from_numpy(p) for p in data[i : i + batch_size]])
            y = torch.ones((x.size(0),), dtype=torch.long) * label
            state = optimizer.update(state, (x.to(device), y.to(device)), inplace=False)

    return state


@torch.inference_mode
def test_for_episode(model, params, data, label, device, batch_size=100):
    num_correct = 0
    for i in range(0, len(data), batch_size):
        x = torch.stack([torch.from_numpy(p) for p in data[i : i + batch_size]])
        outputs = func.functional_call(model, params, x.to(device))
        outputs = outputs.argmax(-1)
        num_correct += len(torch.where(outputs == label)[0])

    return num_correct / len(data)
