import torch
from tqdm import tqdm
import posteriors
import torchopt

from modules.arch import SimpleConvNet
from modules.data.mnist import load, episode_splits
from modules.episode import train_for_episode, test_for_episode
from modules.loss import log_posterior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

x_train, t_train, x_test, t_test = load()
train_set = episode_splits(x_train, t_train)
test_set = episode_splits(x_test, t_test)


def simulate(model, data, batch_size=100, epochs_per_episode=1):
    train_set, test_set = data
    params = dict(model.named_parameters())
    episodes = list(train_set.keys())

    test_metrics = []
    for episode in tqdm(episodes):
        label = episode
        episode_train_data = train_set[episode]
        optimizer = posteriors.torchopt.build(
            log_posterior(model, len(episode_train_data)),
            torchopt.adam(lr=1e-3, maximize=True),
        )
        state = train_for_episode(
            params,
            optimizer,
            episode_train_data,
            label,
            device,
            epochs=epochs_per_episode,
            batch_size=batch_size,
        )
        params = state.params

        step_i_test_metrics = {}
        for prev_episodes in episodes:
            episode_test_data = test_set[prev_episodes]
            accuracy = test_for_episode(
                model,
                params,
                episode_test_data,
                prev_episodes,
                device,
                batch_size=batch_size,
            )
            step_i_test_metrics[prev_episodes] = accuracy

        test_metrics.append(step_i_test_metrics)

    return test_metrics


if __name__ == "__main__":
    model = SimpleConvNet().to(device)

    test_metrics = simulate(
        model, (train_set, test_set), batch_size=10, epochs_per_episode=1
    )

    for i, metrics in enumerate(test_metrics):
        print(f"Episode {i}: {metrics}")
