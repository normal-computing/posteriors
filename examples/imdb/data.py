import torch
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


def load_imdb_dataset(batch_size=32, max_features=20000, max_len=100):
    # Load and pad IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    # Convert data to PyTorch tensors
    train_data = torch.tensor(x_train, dtype=torch.long)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_data = torch.tensor(x_test, dtype=torch.long)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    # Create Tensor datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
