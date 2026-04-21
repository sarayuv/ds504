# test_model.py

import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from model_v2 import SiameseLSTMAttention

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.pairs[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    pairs = data['pairs']
    labels = data['labels']
    return pairs, labels

def test_model(test_dir):
    """
    Initiate the model testing process, including:
    - Loading the saved model
    - Loading and preprocessing test data from test_dir
    - Creating a DataLoader for testing
    - Evaluating the model and printing results
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_file_pattern = os.path.join(test_dir, "X_Y_test100_pairs.pkl")
    
    # Load test data
    test_pairs, test_labels = load_data(test_file_pattern)
    
    test_dataset = SiameseDataset(test_pairs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get the device
    model = SiameseLSTMAttention(input_dim=6, hidden_dim=64).to(device)
    model.load_state_dict(torch.load('best_model_v2.pt', map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = (output > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    test_accu = correct / total

    # Print the accuracy in the required format
    print(f"Accuracy={test_accu:.4f}")

if __name__ == '__main__':
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else 'validation_test'
    test_model(test_dir)