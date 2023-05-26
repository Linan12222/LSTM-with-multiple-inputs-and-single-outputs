import torch
import torch.nn as nn

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        return output, h


class LSTMMain(nn.Module):
    def __init__(self, input_size, output_len, lstm_hidden, lstm_layers, batch_size, device="cpu"):
        super(LSTMMain, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstmunit = LSTM(input_size, lstm_hidden, lstm_layers, batch_size, device)
        self.linear = nn.Linear(lstm_hidden, output_len)

    def forward(self, input_seq):
        ula, h_out = self.lstmunit(input_seq)
        out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
        out = self.linear(out)
        out = out.view(ula.shape[0], ula.shape[1], -1)
        out = out[:, -1, :]
        return out

