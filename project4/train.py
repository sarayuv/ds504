import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import SiameseLSTM
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs  # pairs is a numpy array of shape (n_samples, 2, seq_len, features)
        self.labels = labels  # labels is a numpy array of shape (n_samples,)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.pairs[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def load_data(filename='X_Y_train400_pairs.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    pairs = data['pairs']
    labels = data['labels']
    return pairs, labels

def train(model, optimizer, criterion, train_loader):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################
    return train_loss, train_acc

def evaluate(model, criterion, loader):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################
    return test_loss, test_acc

def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################
