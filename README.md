# Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks

[![DOI](https://zenodo.org/badge/435158242.svg)](https://zenodo.org/badge/latestdoi/435158242)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/snu-mllab/preemptive-robustification/blob/main/LICENSE)

This is the code for reproducing the results of the paper [Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks](https://arxiv.org/abs/2112.05634) accepted at AAAI 2022.

## Requirements

All Python packages required are listed in `requirements.txt`. To install these packages, run the following commands.  

```bash
conda create -n preempt-robust python=3.7
conda activate preempt-robust
pip install -r requirements.txt
```

## Preparing CIFAR-10 data

Download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place it a directory ```./data```.


## Pretrained models

We provide pre-trained checkpoints for adversarially trained model and preemptively robust model.

- ```adv_l2```: ℓ<sub>2</sub> adversarially trained model with early stopping
- ```adv_linf```: ℓ<sub>∞</sub>  adversarially trained model with early stopping
- ```preempt_robust_l2```: ℓ<sub>2</sub> preemptively robust model
- ```preempt_robust_linf```: ℓ<sub>∞</sub> preemptively robust model

We also provide a pre-trained checkpoint for a model with randomized smoothing.
- ```gaussian_0.1```: model trained with additive Gaussian noises (σ = 0.1)

Shell scripts for downloading these checkpoint are located in ```./checkpoints/cifar10/wideresent/[train_type]/download.sh```. You can run each script to download a checkpoint named ```ckpt.pt```. To download all the checkpoints, run ```download_all_ckpts.sh```. You can delete all the checkpoints by running ```delete_all_ckpts.sh```.


## Preemptively robust training

To train preemptively robust classifiers, run the following commands.

### 1. ℓ<sub>2</sub> threat model, ε = δ = 0.5

```bash
python train.py --config ./configs/cifar10_l2_model.yaml
```

### 2. ℓ<sub>∞</sub> threat model, ε = δ = 8/255

```bash
python train.py --config ./configs/cifar10_linf_model.yaml
```

## Preemptive robustification and reconstruction algorithms

To generate preepmtive roobust images and their reconstruction, run the following commands. You can specify the classifier used for generating preemptively robust images by changing ```train_type``` in each yaml file.

### 1. ℓ<sub>2</sub> threat model, ε = δ = 0.5

```bash
python robustify.py --config ./configs/cifar10_l2.yaml
python reconstruct.py --config ./configs/cifar10_l2.yaml
```

### 2. ℓ<sub>∞</sub> threat model, ε = δ = 8/255

```bash
python robustify.py --config ./configs/cifar10_linf.yaml
python reconstruct.py --config ./configs/cifar10_linf.yaml
```

### 3. ℓ<sub>2</sub> threat model, smoothed network, ε = δ = 0.5

```bash
python robustify.py --config ./configs/cifar10_l2_rand.yaml
python reconstruct.py --config ./configs/cifar10_l2_rand.yaml
```
## Grey-box attacks on preemptively robustified images

To conduct grey-box attacks on preemptively robustified images, run the following commands. You can specify attack type by changing ```attack_type_eval``` in each yaml file.

### 1. ℓ<sub>2</sub> threat model, ε = δ = 0.5

```bash
python attack_grey_box.py --config ./configs/cifar10_l2.yaml
```

### 2. ℓ<sub>∞</sub> threat model, ε = δ = 8/255

```bash
python attack_grey_box.py --config ./configs/cifar10_linf.yaml
```

### 3. ℓ<sub>2</sub> threat model, smoothed network, ε = δ = 0.5

```bash
python attack_grey_box.py --config ./configs/cifar10_l2_rand.yaml
```

## White-box attacks on preemptively robustified images  

To conduct white-box attacks on preemptively robustified images, run the following commands. You can specify attack type and its perturbation size by changing ```attack_type_eval``` and ```wbox_epsilon_p``` in each yaml file.

### 1. ℓ<sub>2</sub> threat model, ε = δ = 0.5

```bash
python attack_white_box.py --config ./configs/cifar10_l2.yaml
```

### 2. ℓ<sub>∞</sub> threat model, ε = δ = 8/255

```bash
python attack_white_box.py --config ./configs/cifar10_linf.yaml
```  

### 3. ℓ<sub>2</sub> threat model, smoothed network, ε = δ = 0.5

```bash
python attack_white_box.py --config ./configs/cifar10_l2_rand.yaml
```

## Acknowledgements

This work was supported in part by SNU-NAVER Hyperscale AI Center and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2020-0-00882, (SW STAR LAB) Development of deployable learning intelligence via self-sustainable and trustworthy machine learning and No. 2019-0-01371, Development of brain-inspired AI with human-like intelligence). This material is based upon work supported by the Air Force Office of Scientific Research under award number FA2386-20-1-4043.
