import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from model import SiameseLSTMAttention


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_file = os.path.join(test_dir, "X_Y_test100_pairs.pkl")
    test_pairs, test_labels = load_data(test_file)

    test_dataset = SiameseDataset(test_pairs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SiameseLSTMAttention(input_dim=6, hidden_dim=64).to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
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

    test_acc = correct / total
    print(f"Accuracy={test_acc:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test Siamese Network")
    parser.add_argument("--test_dir", default="./test_data",
                        help="Path to test data directory")
    args = parser.parse_args()
    test_model(args.test_dir)