import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, states):
        h_prev, c_prev = states
        combined = torch.cat((input, h_prev), 1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(self.output_gate(combined))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            initial_h = torch.zeros(
                1,
                input.size(0 if self.batch_first else 1),
                self.hidden_size,
                device=input.device,
            )
            initial_c = torch.zeros(
                1,
                input.size(0 if self.batch_first else 1),
                self.hidden_size,
                device=input.device,
            )
        else:
            initial_h, initial_c = initial_states

        # Ensure we are working with single layer, single direction states
        initial_h = initial_h.squeeze(0)
        initial_c = initial_c.squeeze(0)

        if self.batch_first:
            input = input.transpose(
                0, 1
            )  # Convert (batch, seq_len, feature) to (seq_len, batch, feature)

        outputs = []
        h_t, c_t = initial_h, initial_c

        for i in range(
            input.shape[0]
        ):  # input is expected to be (seq_len, batch, input_size)
            h_t, c_t = self.lstm_cell(input[i], (h_t, c_t))
            outputs.append(h_t.unsqueeze(0))

        outputs = torch.cat(outputs, 0)

        if self.batch_first:
            outputs = outputs.transpose(
                0, 1
            )  # Convert back to (batch, seq_len, feature)

        return outputs, (h_t, c_t)


# Test equivalence
def test_lstm_equivalence():
    input_size = 10
    hidden_size = 20
    seq_len = 5
    batch_size = 1

    # Initialize both LSTMs
    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    custom_lstm = CustomLSTM(input_size, hidden_size, batch_first=True)

    # Manually setting the same weights and biases
    with torch.no_grad():
        # Copy weights and biases from torch LSTM to custom LSTM
        gates = ["input_gate", "forget_gate", "cell_gate", "output_gate"]

        for idx, gate in enumerate(gates):
            start = idx * hidden_size
            end = (idx + 1) * hidden_size

            getattr(custom_lstm.lstm_cell, gate).weight.data[:, :input_size].copy_(
                torch_lstm.weight_ih_l0[start:end]
            )
            getattr(custom_lstm.lstm_cell, gate).weight.data[:, input_size:].copy_(
                torch_lstm.weight_hh_l0[start:end]
            )
            getattr(custom_lstm.lstm_cell, gate).bias.data.copy_(
                torch_lstm.bias_ih_l0[start:end] + torch_lstm.bias_hh_l0[start:end]
            )
    # Dummy input
    inputs = torch.randn(batch_size, seq_len, input_size)

    # Custom LSTM forward pass
    custom_outputs, (custom_hn, custom_cn) = custom_lstm(inputs)

    # Torch LSTM forward pass
    torch_outputs, (torch_hn, torch_cn) = torch_lstm(inputs)

    # Check outputs and final hidden and cell states
    assert torch.allclose(custom_outputs, torch_outputs, atol=1e-6), "Output mismatch"
    assert torch.allclose(custom_hn, torch_hn), "Hidden state mismatch"
    assert torch.allclose(custom_cn, torch_cn), "Cell state mismatch"

    print("Test passed: Custom LSTM and torch.nn.LSTM outputs are equivalent!")


if __name__ == "__main__":
    test_lstm_equivalence()
