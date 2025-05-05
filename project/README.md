# TWM project

## Prerequisites

- Python 3.13
- [pdm](https://pdm-project.org/en/latest/)
  - `pipx install pdm`
- [just](https://just.systems/man/en/)
  - `pipx install rust-just`
- [gdown](https://github.com/wkentaro/gdown)
  - `pipx install gdown`
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

### Train on foggy data

Run the command on forked deeplab repo to train with foggy data:
```shell
pdm run python3 twm/external/deeplab_forked/main.py --data_root "data/cityscapes_foggy" --dataset "cityscapes"\
 --ckpt "checkpoints/deeplabv3plus_mobilenet_cityscapes.pth" --continue_training
```

### Predict example image (saves result to folder test_results)
```shell
pdm run python3 twm/external/deeplab_forked/predict.py --input data/cityscapes_foggy/leftImg8bit/val/frankfurt/frankfurt_000001_032711_leftImg8bit_foggy_beta_0.01.png \
--dataset cityscapes --model deeplabv3plus_mobilenet --ckpt "checkpoints/deeplabv3plus_mobilenet_cityscapes.pth" --save_val_results_to test_results
```

## Running DehazeFormer model

### Train from scratch on foggy data

```shell
pdm run python3 twm/external/dehazeformer_forked/train.py \
  --model dehazeformer-t \
  --save_dir checkpoints \
  --data_dir data/ \
  --dataset cityscapes_foggy \
  --config twm/external/dehazeformer_forked/configs/outdoor/dehazeformer-t.json
```

### Finetune on foggy data (not yet implemented)

```shell
pdm run python3 twm/external/dehazeformer_forked/train.py \
  --model dehazeformer-t \
  --save_dir checkpoints \
  --ckpt checkpoints/dehazeformer-t.pth \
  --data_dir data/ \
  --dataset cityscapes_foggy \
  --config twm/external/dehazeformer_forked/configs/outdoor/dehazeformer-t.json
```

Note: If you encounter `OutOfMemoryError`, try following solutions:
- Set number of workers with argument e.g. `--num_workers 4` (default value is 16). For optimal performance set it to number of CPU cores
- Lower `batch_size` or `patch_size` value in the config file:
`project/twm/external/dehazeformer_forked/configs/outdoor/dehazeformer-t.json`.
- Add `--gpu` argument to ensure model runs on GPU
- Set the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This may reduce VRAM fragmentation and improve memory efficiency

### Evaluation
Saves the results to results/cityscapes_foggy/dehazeformer-t

```
mkdir -p results

pdm run python3 twm/external/dehazeformer_forked/test.py \
  --model dehazeformer-t \
  --ckpt checkpoints/outdoor/dehazeformer-t.pth \
  --data_dir data/ \
  --dataset cityscapes_foggy \
  --config twm/external/dehazeformer_forked/configs/outdoor/dehazeformer-t.json
```

# Usefull commands:

Updating submodules to remote branch (submodule update)
```shell
git submodule update --remote --merge
```
