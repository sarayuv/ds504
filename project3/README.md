# Project 3: MNIST GAN Experiments

This project trains a GAN to generate MNIST digits and saves artifacts for each experiment run.

## Repository Layout

- `gan.py`: Configurable GAN architecture and optimization components
- `train.py`: Experiment runner with logging and artifact generation
- `evaluation.py`: Final image selection and digit coverage checks
- `experiments.py`: Curated batch of three experiments
- `job.sh`: Slurm job script
- `experiments/`: All run outputs and metadata

## Running Experiments

### Single local experiment

```bash
python train.py --experiment-name my_run
```

### Curated three-experiment batch

```bash
python experiments.py
```

## Report Artifacts

Each run creates a folder under `experiments/<timestamp>_<experiment_name>/` with:

- `experiment_config.json`: Full configuration details.
- `epoch_metrics.csv`: Epoch-wise generator/discriminator losses.
- `training_loss.png`: Loss curves.
- `generated_images_5x5.png`: Final 25-image grid.
- `digit_coverage.json`: Coverage of digits 0-9 in the final grid.
- `samples/`: Epoch-by-epoch generated sample grids.
- `generator_best_state_dict.pt`, `generator_state_dict.pt`, `generator.pt`.

A cross-run summary table is maintained at `experiments/experiment_log.csv`.
