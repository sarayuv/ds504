import argparse
import csv
import copy
import json
import os
import random
import shutil
from datetime import datetime
from typing import Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from evaluation import generate_digit_grid
from gan import DEVICE, GANConfig, build_gan_components, build_train_loader, config_to_dict


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# defines and parses command-line arguments for training
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST GAN with report-ready logging.")

    parser.add_argument("--experiment-name", type=str, default="baseline_dcgan")
    parser.add_argument("--notes", type=str, default="")

    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=100)

    parser.add_argument("--g-proj-ch", type=int, default=128)
    parser.add_argument("--g-mid-ch", type=int, default=64)
    parser.add_argument("--d-ch1", type=int, default=64)
    parser.add_argument("--d-ch2", type=int, default=128)
    parser.add_argument("--d-dropout", type=float, default=0.2)

    parser.add_argument("--g-activation", type=str, default="relu", choices=["relu", "leaky_relu", "elu"])
    parser.add_argument("--d-activation", type=str, default="leaky_relu", choices=["relu", "leaky_relu", "elu"])

    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop"])

    parser.add_argument("--loss", type=str, default="bce_logits", choices=["bce_logits"])
    parser.add_argument("--weight-init", type=str, default="dcgan", choices=["dcgan", "xavier", "kaiming"])

    parser.add_argument("--real-label", type=float, default=0.9)
    parser.add_argument("--fake-label", type=float, default=0.0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--d-updates", type=int, default=1)
    parser.add_argument("--g-updates", type=int, default=1)

    parser.add_argument("--seed", type=int, default=504)
    parser.add_argument("--debug-batch-interval", type=int, default=100)

    parser.add_argument("--data-root", type=str, default=os.path.join(PROJECT_DIR, "data"))
    parser.add_argument("--output-root", type=str, default=PROJECT_DIR)

    parser.add_argument("--classifier-epochs", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-max-train-batches", type=int, default=250)
    parser.add_argument("--selection-pool-size", type=int, default=8000)
    parser.add_argument("--selection-batch-size", type=int, default=256)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# converts parsed arguments into a GANConfig dataclass instance
def build_config(args: argparse.Namespace) -> GANConfig:
    return GANConfig(
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        g_proj_channels=args.g_proj_ch,
        g_mid_channels=args.g_mid_ch,
        d_channels1=args.d_ch1,
        d_channels2=args.d_ch2,
        d_dropout=args.d_dropout,
        g_activation=args.g_activation,
        d_activation=args.d_activation,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        optimizer=args.optimizer,
        loss=args.loss,
        weight_init=args.weight_init,
    )


# creates directories for storing experiment outputs and returns paths for the run and samples
def make_run_dirs(output_root: str, experiment_name: str) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{experiment_name}"

    experiments_root = os.path.join(output_root, "experiments")
    run_dir = os.path.join(experiments_root, run_name)
    samples_dir = os.path.join(run_dir, "samples")

    os.makedirs(samples_dir, exist_ok=True)

    return {
        "experiments_root": experiments_root,
        "run_dir": run_dir,
        "samples_dir": samples_dir,
        "run_name": run_name,
    }


# adds noise to labels for training stability
def noisy_labels(base: torch.Tensor, noise: float) -> torch.Tensor:
    if noise <= 0.0:
        return base
    jitter = (2.0 * torch.rand_like(base) - 1.0) * noise
    return torch.clamp(base + jitter, 0.0, 1.0)


# saves generator and discrimintor loss curves as a plot
def save_training_losses(g_losses: List[float], d_losses: List[float], output_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Generator Loss", color="blue")
    plt.plot(d_losses, label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# writes epoch metrics to a CSV file
def write_epoch_metrics_csv(epoch_metrics: List[Dict[str, float]], output_path: str) -> None:
    fieldnames = ["epoch", "avg_d_loss", "avg_g_loss", "avg_d_real_prob", "avg_d_fake_prob"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_metrics)


# appends a summary of the experiment to a CSV file
def append_experiment_summary(summary_path: str, summary_row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    file_exists = os.path.exists(summary_path)

    fieldnames = list(summary_row.keys())
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_row)


# training function
def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_paths = make_run_dirs(args.output_root, args.experiment_name)
    config = build_config(args)

    # prepare dataset and dataloader
    train_loader = build_train_loader(config.batch_size, data_root=args.data_root)
    net_g, net_d, criterion, optimizer_g, optimizer_d = build_gan_components(config, device=DEVICE)

    fixed_noise = torch.randn(64, config.z_dim, device=DEVICE)
    g_losses: List[float] = []
    d_losses: List[float] = []
    epoch_metrics: List[Dict[str, float]] = []
    best_g_loss = float("inf")
    best_g_state_dict = None

    metadata = {
        "experiment_name": args.experiment_name,
        "run_name": run_paths["run_name"],
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "epochs": args.epochs,
        "real_label": args.real_label,
        "fake_label": args.fake_label,
        "label_noise": args.label_noise,
        "d_updates": args.d_updates,
        "g_updates": args.g_updates,
        "debug_batch_interval": args.debug_batch_interval,
        "notes": args.notes,
        "gan_config": config_to_dict(config),
    }

    metadata_path = os.path.join(run_paths["run_dir"], "experiment_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Starting training")
    print(f"Run directory: {run_paths['run_dir']}")
    print(f"Epochs: {args.epochs}, Batches per epoch: {len(train_loader)}")

    # training loop: update discriminator and generator for each batch
    for epoch in range(args.epochs):
        net_g.train()
        net_d.train()

        epoch_d_loss_sum = 0.0
        epoch_g_loss_sum = 0.0
        epoch_d_real_prob_sum = 0.0
        epoch_d_fake_prob_sum = 0.0
        d_steps = 0
        g_steps = 0

        print(f"Epoch {epoch + 1}/{args.epochs} started")

        # iterate through batches of real images, update discriminator and generator, and log metrics
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            # track latest losses for logging
            latest_d_loss = None
            latest_g_loss = None

            # update discriminator for specified number of steps
            for _ in range(args.d_updates):
                # generate fake images with the generator
                noise = torch.randn(batch_size, config.z_dim, device=DEVICE)
                fake_images = net_g(noise).detach()

                # create noisy labels for real and fake images
                real_targets = noisy_labels(
                    torch.full((batch_size, 1), args.real_label, device=DEVICE),
                    args.label_noise,
                )
                fake_targets = noisy_labels(
                    torch.full((batch_size, 1), args.fake_label, device=DEVICE),
                    args.label_noise,
                )

                # compute discriminator loss on real and fake images
                optimizer_d.zero_grad(set_to_none=True)

                # forward pass real and fake images through the discriminator and compute losses
                output_real = net_d(real_images)
                output_fake = net_d(fake_images)

                # compute binary cross-entropy loss for real and fake images
                loss_d_real = criterion(output_real, real_targets)
                loss_d_fake = criterion(output_fake, fake_targets)
                # compute sum of real and fake losses for total discriminator loss
                loss_d = loss_d_real + loss_d_fake

                # backpropagate discriminator loss and update weights
                loss_d.backward()
                optimizer_d.step()

                # compute probabilities of real and fake images being classified as real
                d_prob_real = torch.sigmoid(output_real).mean().item()
                d_prob_fake = torch.sigmoid(output_fake).mean().item()

                # accumulate losses and probabilities for the epoch and log latest discriminator loss
                epoch_d_loss_sum += loss_d.item()
                epoch_d_real_prob_sum += d_prob_real
                epoch_d_fake_prob_sum += d_prob_fake
                d_steps += 1
                latest_d_loss = loss_d.item()

            # update generator for specified number of steps
            for _ in range(args.g_updates):
                # generate fake images and create noisy real labels
                noise = torch.randn(batch_size, config.z_dim, device=DEVICE)
                fake_images = net_g(noise)
                real_targets_for_g = noisy_labels(
                    torch.full((batch_size, 1), args.real_label, device=DEVICE),
                    args.label_noise,
                )

                # compute generator loss based on discriminator's classification of the fake images
                optimizer_g.zero_grad(set_to_none=True)
                output_fake_for_g = net_d(fake_images)
                loss_g = criterion(output_fake_for_g, real_targets_for_g)
                loss_g.backward()
                optimizer_g.step()

                # accumulate generator loss for the epoch
                epoch_g_loss_sum += loss_g.item()
                g_steps += 1
                latest_g_loss = loss_g.item()

            # log metrics at specified intervals
            if batch_idx == 0:
                print(
                    "First batch shapes - "
                    f"real: {tuple(real_images.shape)}, fake: {tuple(fake_images.shape)}"
                )
            
            if (batch_idx + 1) % args.debug_batch_interval == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"D Loss: {latest_d_loss:.4f} | G Loss: {latest_g_loss:.4f}"
                )

        # compute average losses and probabilities for the epoch and log
        avg_d_loss = epoch_d_loss_sum / max(d_steps, 1)
        avg_g_loss = epoch_g_loss_sum / max(g_steps, 1)
        avg_d_real_prob = epoch_d_real_prob_sum / max(d_steps, 1)
        avg_d_fake_prob = epoch_d_fake_prob_sum / max(d_steps, 1)

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "avg_d_loss": avg_d_loss,
                "avg_g_loss": avg_g_loss,
                "avg_d_real_prob": avg_d_real_prob,
                "avg_d_fake_prob": avg_d_fake_prob,
            }
        )

        # generate sample images from the generator at the end of the epoch and save
        net_g.eval()
        with torch.no_grad():
            samples = net_g(fixed_noise)
            samples = (samples + 1.0) / 2.0
            sample_grid = make_grid(samples, nrow=8, padding=2)
            save_image(sample_grid, os.path.join(run_paths["samples_dir"], f"epoch_{epoch + 1:03d}.png"))

        # save the best generator model based on lowest average generator loss
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_g_state_dict = copy.deepcopy(net_g.state_dict())
            torch.save(best_g_state_dict, os.path.join(run_paths["run_dir"], "generator_best_state_dict.pt"))
            torch.save(best_g_state_dict, os.path.join(PROJECT_DIR, "generator_best_state_dict.pt"))

        print(
            f"Epoch {epoch + 1}/{args.epochs} complete | "
            f"Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f} | "
            f"D(real): {avg_d_real_prob:.3f} | D(fake): {avg_d_fake_prob:.3f}"
        )

    print("Training complete")

    # export the best generator for submission
    if best_g_state_dict is None:
        best_g_state_dict = copy.deepcopy(net_g.state_dict())
    net_g.load_state_dict(best_g_state_dict)

    torch.save(net_g, os.path.join(run_paths["run_dir"], "generator.pt"))
    torch.save(best_g_state_dict, os.path.join(run_paths["run_dir"], "generator_weights.pt"))
    torch.save(net_g, os.path.join(PROJECT_DIR, "generator.pt"))
    torch.save(best_g_state_dict, os.path.join(PROJECT_DIR, "generator_weights.pt"))

    losses_path = os.path.join(run_paths["run_dir"], "training_loss.png")
    save_training_losses(g_losses, d_losses, losses_path)
    shutil.copy2(losses_path, os.path.join(PROJECT_DIR, "training_loss.png"))

    metrics_csv_path = os.path.join(run_paths["run_dir"], "epoch_metrics.csv")
    write_epoch_metrics_csv(epoch_metrics, metrics_csv_path)

    # generate grid of generated images and save artifacts
    coverage_result = generate_digit_grid(
        generator=net_g,
        z_dim=config.z_dim,
        device=DEVICE,
        output_dir=run_paths["run_dir"],
        data_root=args.data_root,
        seed=args.seed,
        pool_size=args.selection_pool_size,
        pool_batch_size=args.selection_batch_size,
        classifier_epochs=args.classifier_epochs,
        classifier_batch_size=args.classifier_batch_size,
        classifier_max_train_batches=args.classifier_max_train_batches,
    )

    coverage_grid_path = cast(str, coverage_result["grid_path"])
    coverage_metadata = cast(Dict[str, object], coverage_result["metadata"])
    digit_coverage = cast(Dict[str, int], coverage_metadata["digit_coverage"])
    missing_digits = cast(List[int], coverage_metadata["missing_digits"])

    shutil.copy2(coverage_grid_path, os.path.join(PROJECT_DIR, "generated_images.png"))

    # compile a summary of the experiment run and save
    run_summary = {
        "run_name": run_paths["run_name"],
        "experiment_name": args.experiment_name,
        "epochs": args.epochs,
        "batch_size": config.batch_size,
        "z_dim": config.z_dim,
        "g_proj_channels": config.g_proj_channels,
        "g_mid_channels": config.g_mid_channels,
        "d_channels1": config.d_channels1,
        "d_channels2": config.d_channels2,
        "d_dropout": config.d_dropout,
        "g_activation": config.g_activation,
        "d_activation": config.d_activation,
        "weight_init": config.weight_init,
        "loss": config.loss,
        "optimizer": config.optimizer,
        "g_lr": config.g_lr,
        "d_lr": config.d_lr,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "real_label": args.real_label,
        "fake_label": args.fake_label,
        "label_noise": args.label_noise,
        "d_updates": args.d_updates,
        "g_updates": args.g_updates,
        "final_d_loss": d_losses[-1],
        "final_g_loss": g_losses[-1],
        "best_g_loss": min(g_losses),
        "digit_coverage": json.dumps(digit_coverage),
        "missing_digits": json.dumps(missing_digits),
        "run_dir": run_paths["run_dir"],
    }

    summary_csv = os.path.join(run_paths["experiments_root"], "experiment_log.csv")
    append_experiment_summary(summary_csv, run_summary)

    final_report = {
        "config": metadata,
        "run_summary": run_summary,
        "artifacts": {
            "run_dir": run_paths["run_dir"],
            "samples_dir": run_paths["samples_dir"],
            "loss_plot": losses_path,
            "final_generated_grid": coverage_result["grid_path"],
            "digit_coverage_json": coverage_result["metadata_path"],
            "epoch_metrics_csv": metrics_csv_path,
            "generator_pt": os.path.join(run_paths["run_dir"], "generator.pt"),
            "generator_weights": os.path.join(run_paths["run_dir"], "generator_weights.pt"),
            "generator_best_state_dict": os.path.join(run_paths["run_dir"], "generator_best_state_dict.pt"),
        },
    }

    final_report_path = os.path.join(run_paths["run_dir"], "final_report_metadata.json")
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("Saved project-level artifacts:")
    print(f"- {os.path.join(PROJECT_DIR, 'generated_images.png')}")
    print(f"- {os.path.join(PROJECT_DIR, 'training_loss.png')}")
    print(f"- {os.path.join(PROJECT_DIR, 'generator.pt')}")
    print(f"- {os.path.join(PROJECT_DIR, 'generator_weights.pt')}")

    print("Saved run-level artifacts:")
    print(f"- {run_paths['run_dir']}")
    print(f"- {final_report_path}")


if __name__ == "__main__":
    train()
