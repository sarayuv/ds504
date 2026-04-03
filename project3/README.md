# Project 3: MNIST GAN Experiments

This project trains a GAN to generate MNIST digits and saves report-ready artifacts for each experiment run.

The training pipeline is designed to support your report requirements:

- Reproducible experiment configurations
- Model/hyperparameter tracking
- Loss curves for generator and discriminator
- Final 5x5 generated image grid
- Digit coverage metadata for final generated images

## Repository Layout

- `gan.py`: configurable GAN architecture and optimization components
- `train.py`: experiment runner with logging and artifact generation
- `evaluation.py`: report-ready final image selection and digit coverage checks
- `experiments.py`: curated batch of three experiments (submitted through Slurm)
- `job.sh`: Slurm job script
- `experiments/`: all run outputs and metadata
- `samples/`: legacy/compatibility sample outputs

## Environment Setup

Activate your environment:

```bash
conda activate project3_env
```

## Running Experiments

### 1) Single local experiment

```bash
cd /home/svijayanagaram/ds504/project3
python train.py --experiment-name my_run
```

### 2) Single Slurm experiment

From `project3` directory:

```bash
cd /home/svijayanagaram/ds504/project3
TRAIN_ARGS="--experiment-name my_slurm_run --epochs 50" sbatch job.sh
```

### 3) Curated three-experiment batch

This submits the three best-performing style experiments from your earlier results:

```bash
cd /home/svijayanagaram/ds504/project3
python experiments.py
```

## Report Artifacts (Per Run)

Each run creates a folder under `experiments/<timestamp>_<experiment_name>/` with:

- `experiment_config.json`
  - architecture choices (layers/channels, activations)
  - optimizer, learning rates, betas
  - loss function and weight initialization
  - training hyperparameters (epochs, batch size, labels, update ratios)
- `epoch_metrics.csv`
  - epoch-wise generator/discriminator losses and discriminator probabilities
- `training_loss.png`
  - generator/discriminator loss curves
- `generated_images_5x5.png`
  - final 25-image grid
- `digit_coverage.json`
  - coverage of digits 0-9 in the selected final grid
- `samples/`
  - epoch-by-epoch generated sample grids
- `generator_best_state_dict.pt`, `generator_state_dict.pt`, `generator.pt`

A cross-run summary table is also maintained at:

- `experiments/experiment_log.csv`

## Mapping to Report Requirements

### Set of experiments performed

Use:

- `experiments/experiment_log.csv` (all runs and key settings)
- each run's `experiment_config.json` (full config)

You can discuss:

- architecture structure: generator/discriminator channels and activations
- hyperparameters: epochs, batch size, learning rates, label settings, update ratios
- optimization choices: optimizer and betas
- initialization choices: dcgan/xavier/kaiming
- loss function: BCE with logits

### Special skills to improve generation quality

You can report and compare:

- model capacity changes (`--g-proj-ch`, `--g-mid-ch`, `--d-ch1`, `--d-ch2`)
- regularization (`--d-dropout`)
- training stability knobs (`--real-label`, `--fake-label`, `--label-noise`, `--d-updates`, `--g-updates`)
- optimizer/lr settings (`--optimizer`, `--g-lr`, `--d-lr`, `--beta1`, `--beta2`)
- initialization (`--weight-init`)

### Visualization requirement

Use per-run:

- `generated_images_5x5.png` for final generated 25 images
- `training_loss.png` for loss plot
- `digit_coverage.json` to verify at least one sample for each digit class in the selected final grid

## Useful Commands

Check queue:

```bash
squeue -u "$USER"
```

Tail latest job logs:

```bash
tail -f project3_output_<jobid>.log
tail -f project3_error_<jobid>.log
```

## Notes

- `job.sh` uses `SLURM_SUBMIT_DIR`, so submit from the `project3` directory for expected behavior.
- Project-level convenience artifacts are also written to:
  - `generated_images.png`
  - `training_loss.png`
  - `generator.pt`, `generator_state_dict.pt`, `generator_best_state_dict.pt`
