# Write your BiLSTMEncoder class here
import torch.nn as nn


class LSTMEncoder(nn.Module):

    def __init__(self, input_channels, hidden_channels, num_layers = 2, dropout = 0.3, bidirectional = True, batch_first= True):

        super().__init__()
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_channels, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional, dropout = dropout)
        self.fc = nn.Linear(hidden_channels * 2, hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x, _ = self.lstm(x)
        return x