import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.nn.LSTM does not work well with torch.func https://github.com/pytorch/pytorch/issues/105982
# so use custom simplfied LSTM instead
from examples.imdb.lstm import CustomLSTM


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        max_features=20000,
        embedding_size=128,
        cell_size=128,
        num_filters=64,
        kernel_size=5,
        pool_size=4,
        use_swish=False,
        use_maxpool=True,
    ):
        super(CNNLSTM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max_features, embedding_dim=embedding_size
        )
        self.conv1d = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )
        self.use_swish = use_swish
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.lstm = CustomLSTM(
            input_size=num_filters, hidden_size=cell_size, batch_first=True
        )

        self.fc = nn.Linear(in_features=cell_size, out_features=num_classes)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # Shape: [batch_size, seq_length, embedding_size]
        x = x.permute(
            0, 2, 1
        )  # Shape: [batch_size, embedding_size, seq_length] to match Conv1d input

        # Convolution
        x = self.conv1d(x)
        if self.use_swish:
            x = x * torch.sigmoid(x)  # Swish activation function
        else:
            x = F.relu(x)

        # Pooling
        if self.use_maxpool:
            x = self.maxpool(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # Shape: [batch_size, seq_length, num_filters]

        # LSTM
        output, _ = self.lstm(x)

        # Take the last sequence output
        last_output = output[:, -1, :]

        # Fully connected layer
        logits = self.fc(last_output)

        return logits
