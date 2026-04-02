import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from gan import (
    BATCH_SIZE,
    DEVICE,
    Z_DIM,
    criterion,
    netD,
    netG,
    optimizerD,
    optimizerG,
    train_loader,
)

# hyperparameters
EPOCHS = 50
REAL_LABEL = 0.9
FAKE_LABEL = 0.0
DEBUG_BATCH_INTERVAL = 100

# Ensure all generated files are saved in the project3 directory.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def train():
    g_losses = []
    d_losses = []

    print(f"Starting training loop")
    print(f"Epochs: {EPOCHS}, Batches per epoch: {len(train_loader)}")

    for epoch in range(EPOCHS):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        print(f"Epoch {epoch + 1}/{EPOCHS} started")

        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.view(BATCH_SIZE, -1).to(DEVICE)

            noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
            fake_images = netG(noise)

            fake_labels = torch.full((BATCH_SIZE, 1), FAKE_LABEL, device=DEVICE)
            real_labels = torch.full((BATCH_SIZE, 1), REAL_LABEL, device=DEVICE)

            # Train discriminator.
            netD.zero_grad()
            output_real = netD(real_images)
            loss_d_real = criterion(output_real, real_labels)

            output_fake = netD(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            # Train generator.
            netG.zero_grad()
            output_fake_for_g = netD(fake_images)
            loss_g = criterion(output_fake_for_g, real_labels)
            loss_g.backward()
            optimizerG.step()

            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()

            if batch_idx == 0:
                print(
                    "First batch shapes - "
                    f"real: {tuple(real_images.shape)}, fake: {tuple(fake_images.shape)}"
                )

            if (batch_idx + 1) % DEBUG_BATCH_INTERVAL == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}"
                )

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} complete | "
            f"Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}"
        )

    print("Training complete")
    return g_losses, d_losses


def save_generator():
    torch.save(netG, os.path.join(PROJECT_DIR, "generator.pt"))
    torch.save(netG.state_dict(), os.path.join(PROJECT_DIR, "generator_state_dict.pt"))
    print("Saved generator to project3 directory")


def plot_generated_images(num_gen=25):
    np.random.seed(504)
    h = w = 28

    netG.eval()
    with torch.no_grad():
        z = torch.tensor(
            np.random.normal(0, 1, (num_gen, Z_DIM)), dtype=torch.float32, device=DEVICE
        )
        generated_images = netG(z).cpu().numpy()

    n = int(np.sqrt(num_gen))
    image_grid = np.empty((h * n, w * n))
    for i in range(n):
        for j in range(n):
            img = generated_images[i * n + j].reshape(h, w)
            img = (img + 1) / 2
            image_grid[i * h:(i + 1) * h, j * w:(j + 1) * w] = img

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title("Generated MNIST Digits")
    plt.imshow(image_grid, cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "generated_images.png"), dpi=150, bbox_inches="tight")
    print("Saved: generated_images.png in project3 directory")
    plt.show()


def plot_training_losses(g_losses, d_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Generator Loss", color="blue")
    plt.plot(d_losses, label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "training_loss.png"), dpi=150)
    print("Saved: training_loss.png in project3 directory")
    plt.show()


if __name__ == "__main__":
    g_losses, d_losses = train()
    save_generator()
    plot_generated_images(num_gen=25)
    plot_training_losses(g_losses, d_losses)