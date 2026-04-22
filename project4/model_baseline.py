import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super(SiameseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        output, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return hidden

    def forward(self, x):
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]
        emb1 = self.encode(x1)
        emb2 = self.encode(x2)
        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, prod], dim=1)
        out = self.fc(combined)
        return out.squeeze(1)