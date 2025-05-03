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

Fetch external repositories:

```shell
just fetch-repos
```

## Running deeplab models

### Train on foggy data
(foggy data inside data folder have to be without surfix, and normal data with some surfix eg. org)

Run the command on forked deeplab repo to train with foggy data:
```shell
python twm/external/deeplab_forked/main.py --data_root "data/cityscapes" --dataset "cityscapes"\
 --ckpt "checkpoint/cityscapes.pth" --continue_training
```

### Predict example image (saves result to folder test_results)
```shell
python twm/external/deeplab_forked/predict.py --input data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_032711_leftImg8bit_foggy_beta_0.01.png \
--dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/cityscapes.pth --save_val_results_to test_results
```
