import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data
from train import TaxiDriverDataset, evaluate

def test_model(test_dir):
    """
    Initiate the model testing process, including:
    - Loading the saved model
    - Loading and preprocessing test data from test_dir
    - Creating a DataLoader for testing
    - Evaluating the model and printing results
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct the file pattern using test_dir
    test_file_pattern = os.path.join(test_dir, "*.csv")
    # Load test data
    X_test, y_test = load_data(test_file_pattern)

    # remap labels to a contiguous range starting from 0
    unique_labels = sorted(set(y_test))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    y_mapped = np.array([label_map[lbl] for lbl in y_test])
    num_classes = len(unique_labels)

    # rebuild model
    input_dim = X_test.shape[2]

    model = TaxiDriverClassifier(input_dim=input_dim, output_dim=num_classes)

    # ensures weights saved on GPU load correctly on CPU
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model = model.to(device)

    # create test dataset and dataloader
    test_dataset = TaxiDriverDataset(
        tensors=(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_mapped, dtype=torch.long),
        )
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    _test_loss, test_accu = evaluate(model, criterion, test_loader, device)

    # Print the accuracy in the required format
    print(f"Accuracy={test_accu:.4f}")