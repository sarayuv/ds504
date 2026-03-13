# train.py

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier
from extract_feature import load_data

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

class TaxiDriverDataset(Dataset):
    """
    Custom dataset class for Taxi Driver Classification.
    Handles loading and preparing data for the model
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform


    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, idx):
        x = self.tensors[0][idx]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][idx]
        return x, y

def train(model, optimizer, criterion, train_loader, device, epoch):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """

    # enable dropout during training
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    running_loss = 0.0

    # loop over batches of data
    for i, (inp, lab) in enumerate(train_loader, 0):
        # get the inputs- data is a list of [inputs, labels]
        inputs, labels = inp.to(device), lab.to(device)
        
        optimizer.zero_grad()

        # forward pass
        logits = model(inputs)
        # compute loss
        loss = criterion(logits, labels)
        # compute gradients
        loss.backward()

        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update weights
        optimizer.step()

        # accumulate loss and compute accuracy
        total_loss += loss.item()
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # print statistics every 100 batches
        print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / (i + 1)))
    
    # average loss and accuracy for the epoch
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc

# Define the testing function
def evaluate(model, criterion, test_loader, device):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    
    # disable dropout during evaluation
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # no gradient computation needed during evaluation
    with torch.no_grad():
        for i, (inp, lab) in enumerate(test_loader, 0):
            # get the inputs- data is a list of [inputs, labels]
            inputs, labels = inp.to(device), lab.to(device)

            # forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)

            # accumulate loss
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    # average loss
    test_loss = total_loss / len(test_loader)
    # compute accuracy
    test_acc = correct / total

    return test_loss, test_acc

def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    # hyperparameters
    NUM_EPOCHS = 80
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    VAL_SPLIT = 0.2
    DATA_PATTERN = "../data/*.csv"
    MODEL_PATH = "best_model.pt"

    # load data
    print("Loading training data")
    X, y = load_data(DATA_PATTERN)

    if X.size == 0:
        raise RuntimeError("No data loaded. Check the data files and pattern.")
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")

    # remap labels
    unique_labels = sorted(np.unique(y))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    y_remapped = np.array([label_map[lbl] for lbl in y])
    num_classes = len(unique_labels)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y_remapped, test_size=VAL_SPLIT, random_state=42, stratify=y_remapped)

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # data loaders
    train_dataset = TaxiDriverDataset(
        tensors=(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
    )
    val_dataset = TaxiDriverDataset(
        tensors=(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model, loss, optimizer, scheduler
    input_dim = X.shape[2]

    model = TaxiDriverClassifier(input_dim=input_dim, output_dim=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # reduce learning rate if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    # training loop
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # train for one epoch
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, device, epoch)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        # step the learning rate scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                },
                MODEL_PATH,
            )
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
        
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")