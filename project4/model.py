
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseLSTM(nn.Module):
    """
    Neural network model for training Siamese Networks.
    """
    def __init__(self, input_dim, hidden_dim):
        super(SiameseLSTM, self).__init__()
        
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # The input dataset shape is (number of trajectory pairs, 2, 100, feature size), split the input into two parts and train in the shared model.
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]

        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        return x
