# TWM project

## Prerequisites

- Python 3.13
- [pdm](https://pdm-project.org/en/latest/)
  - `pipx install pdm`
- [just](https://just.systems/man/en/)
  - `pipx install rust-just`
- [Weights & Biases account](https://wandb.ai/site/)
- [CityScapes account](https://www.cityscapes-dataset.com/)
- ~74GB of free disk space (43GB after downloaded zipfiles are removed)

## Setup

Setup the environment:

```shell
just setup
```

Note: you will be prompted to enter your [wandb token](https://docs.wandb.ai/quickstart/)

Download datasets:

```shell
just get-data
```
Note: you will be prompted to enter your CityScapes credentials

Fetch pre-trained models:

```shell
just get-models
```

## Running deeplab models

### Finetune on foggy data

Run the command on forked deeplab repo to train with foggy data:

```shell
pdm run python3 train_test_deeplab.py \
    --data_root "data/cityscapes_foggy" \
    --dataset "cityscapes" \
    --val_interval 2000 \
    --lr 0.002  \
    --crop_size 768 \
    --batch_size 16 \
    --output_stride 16 \
    --continue_training \
    --ignore_previous_best_score \
    --enable_wandb \
    --wandb_team tomasz-owienko-stud-warsaw-university-of-technology \
    --wandb_project twm \
    --ckpt "checkpoints/deeplabv3plus_mobilenet_cityscapes.pth"
```

If the model cannot fit on your GPU, reduce the batch size. Please do not change `--crop_size` and `--output_stride`, as they strongly affect model preformance.

You can also resume training from a checkpoint saved on Weights & Biases - in that case, do not specify `--ckpt` and provide two parameters:

- `--wandb_restore_run_path` - run path of a run which holds the desired checkpoint
- `--wandb_restore_ckpt` - path to checkpoint relative to run's root

Eg.:

```shell
pdm run python3 train_test_deeplab.py \
  # other args
  --wandb_restore_ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
  --wandb_restore_run_path tomasz-owienko-stud-warsaw-university-of-technology/twm/abcdefgh
```

### Predict masks from example image / all images in specified directory (saves result to folder `test_results`)

```shell
pdm run predict_deeplab.py \
    --input data/cityscapes_foggy/leftImg8bit/val/frankfurt/frankfurt_000001_032711_leftImg8bit_foggy_beta_0.02.png \
    --dataset cityscapes \
    --model deeplabv3plus_mobilenet \
    --save_val_results_to test_results \
    --ckpt checkpoints/deeplabv3plus_mobilenet_cityscapes.pth
```

Again, checkpoint can also be loaded from wandb:

```shell
pdm run python3 predict_deeplab.py \
  # other args
  --wandb_restore_ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
  --wandb_restore_run_path tomasz-owienko-stud-warsaw-university-of-technology/twm/abcdefgh
```

# Useful commands:

Updating submodules to remote branch (submodule update)
```shell
git submodule update --remote --merge
```
