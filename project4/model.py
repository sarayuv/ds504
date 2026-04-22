import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.v(torch.tanh(self.attn(lstm_output))), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


class SiameseLSTMAttention(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super(SiameseLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        lstm_out, _ = self.lstm(x)
        embedding = self.attention(lstm_out)
        return embedding

    def forward(self, x):
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]
        emb1 = self.encode(x1)
        emb2 = self.encode(x2)
        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, prod], dim=1)
        out = self.fc(combined)
        return out.squeeze(1)