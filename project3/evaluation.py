import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _train_digit_classifier(
    model: nn.Module,
    device: torch.device,
    data_root: str,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
) -> nn.Module:
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=_mnist_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if batch_idx + 1 >= max_train_batches:
                break

    return model


def load_or_train_digit_classifier(
    cache_path: str,
    device: torch.device,
    data_root: str,
    epochs: int = 2,
    batch_size: int = 256,
    max_train_batches: int = 250,
) -> nn.Module:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    model = DigitClassifier().to(device)
    if os.path.exists(cache_path):
        model.load_state_dict(torch.load(cache_path, map_location=device))
        model.eval()
        return model

    model = _train_digit_classifier(
        model,
        device=device,
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        max_train_batches=max_train_batches,
    )
    torch.save(model.state_dict(), cache_path)
    model.eval()
    return model


def _sample_and_score(
    generator: nn.Module,
    classifier: nn.Module,
    z_dim: int,
    device: torch.device,
    pool_size: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    generated_batches: List[torch.Tensor] = []
    pred_batches: List[torch.Tensor] = []
    prob_batches: List[torch.Tensor] = []

    remaining = pool_size
    generator.eval()
    classifier.eval()
    with torch.no_grad():
        while remaining > 0:
            current_bs = min(batch_size, remaining)
            noise = torch.randn(current_bs, z_dim, device=device, generator=rng)
            images = generator(noise)
            logits = classifier(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            generated_batches.append(images.cpu())
            pred_batches.append(preds.cpu())
            prob_batches.append(probs.cpu())

            remaining -= current_bs

    generated = torch.cat(generated_batches, dim=0)
    preds = torch.cat(pred_batches, dim=0)
    probs = torch.cat(prob_batches, dim=0)
    return generated, preds, probs


def _select_with_digit_coverage(
    images: torch.Tensor,
    preds: torch.Tensor,
    probs: torch.Tensor,
    total_images: int = 25,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    num_samples = images.size(0)

    selected_indices: List[int] = []
    assigned_digits: List[int] = []
    used = set()

    # Reserve one unique image for each digit 0..9 first.
    for digit in range(10):
        ranked_for_digit = torch.argsort(probs[:, digit], descending=True).tolist()
        chosen = None
        for idx in ranked_for_digit:
            if idx not in used:
                chosen = idx
                break
        if chosen is None:
            chosen = ranked_for_digit[0]

        selected_indices.append(chosen)
        assigned_digits.append(digit)
        used.add(chosen)

    # Fill remaining slots with highest-confidence predictions overall.
    confidence = torch.max(probs, dim=1).values
    ranked_indices = torch.argsort(confidence, descending=True).tolist()

    for idx in ranked_indices:
        if len(selected_indices) >= total_images:
            break
        if idx in used:
            continue
        selected_indices.append(idx)
        used.add(idx)

    selected_indices = selected_indices[:total_images]
    selected = images[selected_indices]

    selected_preds = preds[selected_indices].tolist()
    coverage = {str(d): selected_preds.count(d) for d in range(10)}

    metadata = {
        "selected_indices": selected_indices,
        "slot_digit_assignment": assigned_digits + [None] * max(0, total_images - len(assigned_digits)),
        "selected_pred_digits": selected_preds,
        "digit_coverage": coverage,
        "missing_digits": [d for d, c in coverage.items() if c == 0],
    }
    return selected, metadata


def save_digit_grid(images: torch.Tensor, output_path: str, title: str = "Generated MNIST Digits") -> None:
    image_array = images.numpy()
    image_array = (image_array + 1.0) / 2.0
    image_array = np.clip(image_array, 0.0, 1.0)

    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    fig.suptitle(title, fontsize=16)

    idx = 0
    for r in range(5):
        for c in range(5):
            axes[r, c].imshow(image_array[idx, 0], cmap="gray")
            axes[r, c].axis("off")
            idx += 1

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report_ready_grid(
    generator: nn.Module,
    z_dim: int,
    device: torch.device,
    output_dir: str,
    data_root: str,
    seed: int = 504,
    pool_size: int = 8000,
    pool_batch_size: int = 256,
    classifier_epochs: int = 2,
    classifier_batch_size: int = 256,
    classifier_max_train_batches: int = 250,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = os.path.join(output_dir, "cache")
    classifier_path = os.path.join(cache_dir, "digit_classifier.pt")

    classifier = load_or_train_digit_classifier(
        cache_path=classifier_path,
        device=device,
        data_root=data_root,
        epochs=classifier_epochs,
        batch_size=classifier_batch_size,
        max_train_batches=classifier_max_train_batches,
    )

    images, preds, probs = _sample_and_score(
        generator=generator,
        classifier=classifier,
        z_dim=z_dim,
        device=device,
        pool_size=pool_size,
        batch_size=pool_batch_size,
        seed=seed,
    )

    selected, metadata = _select_with_digit_coverage(images, preds, probs, total_images=25)

    grid_path = os.path.join(output_dir, "generated_images_5x5.png")
    save_digit_grid(selected, grid_path)

    metadata_path = os.path.join(output_dir, "digit_coverage.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "grid_path": grid_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }
