import re
import matplotlib.pyplot as plt


def parse_slurm_log(filepath):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    with open(filepath, 'r') as f:
        content = f.read()

    v2_pattern = r'Epoch (\d+)/25 - Train Loss: ([\d.]+), Train Acc: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+)'

    for match in re.finditer(v2_pattern, content):
        train_loss.append(float(match.group(2)))
        train_acc.append(float(match.group(3)))
        val_loss.append(float(match.group(4)))
        val_acc.append(float(match.group(5)))

    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }


def plot_training_curves(data, output_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(data['train_loss']) + 1)

    axes[0].plot(epochs, data['train_loss'], 'b-o', label='Training Loss', markersize=5)
    axes[0].plot(epochs, data['val_loss'], 'r-s', label='Validation Loss', markersize=5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, data['train_acc'], 'b-o', label='Training Accuracy', markersize=5)
    axes[1].plot(epochs, data['val_acc'], 'r-s', label='Validation Accuracy', markersize=5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


if __name__ == '__main__':
    log_file = 'logs/slurm_project4_1972317.out'
    data = parse_slurm_log(log_file)
    plot_training_curves(data, 'training_curves.png')