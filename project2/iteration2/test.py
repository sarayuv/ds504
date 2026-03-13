import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import TaxiDriverClassifier
from extract_feature import load_data
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

    if X_test.size == 0:
        raise RuntimeError("No test data loaded. Check --test_dir and CSV files.")

    # ensures weights saved on GPU load correctly on CPU
    checkpoint = torch.load("best_model.pt", map_location=device)

    # extract model parameters from checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_dim = int(checkpoint.get("input_dim", X_test.shape[2]))
        num_classes = int(checkpoint.get("num_classes", state_dict["fc2.weight"].shape[0]))
        label_map = checkpoint.get("label_map")
    else:
        # fallback for older checkpoint format where entire model was saved
        state_dict = checkpoint
        input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        num_classes = state_dict["fc2.weight"].shape[0]
        label_map = None

    # map test labels using training label map if available
    if label_map is not None:
        unknown = sorted(set(y_test) - set(label_map.keys()))
        if unknown:
            raise RuntimeError(f"Test set has unseen labels not present in training data: {unknown}")
        y_mapped = np.array([label_map[lbl] for lbl in y_test])
    else:
        # create a new mapping based on test labels
        unique_labels = sorted(set(y_test))
        temp_map = {orig: idx for idx, orig in enumerate(unique_labels)}
        y_mapped = np.array([temp_map[lbl] for lbl in y_test])

    # rebuild model
    model = TaxiDriverClassifier(input_dim=input_dim, output_dim=num_classes)
    model.load_state_dict(state_dict)
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