import subprocess

# define experiments and their command-line arguments for training
EXPERIMENTS = [
    {
        "name": "best_baseline_dcgan",
        "args": [
            "--epochs",
            "50",
            "--batch-size",
            "128",
            "--g-proj-ch",
            "128",
            "--g-mid-ch",
            "64",
            "--d-ch1",
            "64",
            "--d-ch2",
            "128",
            "--d-dropout",
            "0.2",
            "--g-lr",
            "0.0002",
            "--d-lr",
            "0.0002",
            "--weight-init",
            "dcgan",
            "--notes",
            "Best_earlier_baseline_architecture",
        ],
    },
    {
        "name": "best_wider_generator",
        "args": [
            "--epochs",
            "60",
            "--batch-size",
            "128",
            "--g-proj-ch",
            "192",
            "--g-mid-ch",
            "96",
            "--d-ch1",
            "64",
            "--d-ch2",
            "128",
            "--d-dropout",
            "0.2",
            "--g-lr",
            "0.0002",
            "--d-lr",
            "0.0002",
            "--weight-init",
            "dcgan",
            "--notes",
            "Wider_generator_with_balanced_discriminator",
        ],
    },
    {
        "name": "best_longer_training_low_lr",
        "args": [
            "--epochs",
            "80",
            "--batch-size",
            "128",
            "--g-proj-ch",
            "128",
            "--g-mid-ch",
            "64",
            "--d-ch1",
            "64",
            "--d-ch2",
            "128",
            "--d-dropout",
            "0.15",
            "--g-lr",
            "0.00015",
            "--d-lr",
            "0.00015",
            "--weight-init",
            "dcgan",
            "--notes",
            "Longer_training_lower_lr_lighter_dropout",
        ],
    },
]


def run_experiments() -> None:
    for exp in EXPERIMENTS:
        train_args = ["--experiment-name", exp["name"], *exp["args"]]
        train_args_str = " ".join(train_args)
        cmd = [
            "bash",
            "-lc",
            f"cd /home/svijayanagaram/ds504/project3 && TRAIN_ARGS='{train_args_str}' sbatch job.sh",
        ]
        print("Submitting:", train_args_str)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_experiments()
