import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model_baseline import SiameseLSTM
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = (output > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item() * x.size(0)
            pred = (output > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def train_model():
    print(f"\n{'='*50}")
    print("Training baseline model")
    print(f"{'='*50}")

    pairs, labels = load_data('X_Y_train400_pairs.pkl')

    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42
    )

    train_dataset = SiameseDataset(train_pairs, train_labels)
    val_dataset = SiameseDataset(val_pairs, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SiameseLSTM(input_dim=6, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    best_val_acc = 0
    num_epochs = 15

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_acc = evaluate(model, criterion, val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_baseline.pt')
            print(f"  -> Saved best model with val_acc: {val_acc:.4f}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    train_model()