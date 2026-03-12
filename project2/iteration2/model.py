# model.py

import torch.nn as nn
import torch.nn.functional as F

class TaxiDriverClassifier(nn.Module):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    """
    def __init__(self, input_dim, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        # fixed hyperparameters
        hidden_dim = 128  # number of hidden units in LSTM layer
        dropout = 0.4  # dropout rate to prevent overfitting

        # LSTM layer processes sequential data
        # input_dim: number of features in input
        # hidden_size: number of features in hidden state
        # num_layers: number of stacked LSTM layers
        # batch_first: if true, input and output tensors are provided as (batch, seq, feature)
        # bidirectional: if true, LSTM is bidirectional
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        # randomly zero some elements during training to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # fully connected layers to map LSTM output to class logits
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # pass input through LSTM layer
        # lstm_out: output features from the last layer of the LSTM for each time step
        # h_n: hidden state for the last time step (num_layers * num_directions)
        lstm_out, (h_n, _) = self.lstm(x)

        # LSTM summary of the entire trajectory
        lstm_summary = h_n[-1]
        
        # apply dropout to the LSTM summary
        x = self.dropout(lstm_summary)

        # fully connected layers to map LSTM output to class logits
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    