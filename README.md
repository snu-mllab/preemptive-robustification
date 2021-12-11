# Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks

This is the code for reproducing the results of the paper "Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks" accepted at AAAI 2022.

## Requirements

All Python packages required are listed in `requirements.txt`. To install these packages, run the following commands:  

```bash
conda create -n preemp-robust python=3.7
conda activate preemp-robust
pip install -r requirements.txt
```

## Preparing CIFAR-10 data

Download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place it a directory ```./data```.

## Training a preemptively robust classifier

### 1. ℓ<sub>2</sub> threat model, ε = δ = 0.5

Run the following command.
```bash
python train.py --config ./configs/cifar10_l2_model.yaml
```

### 2. ℓ<sub>∞</sub> threat model, ε = δ = 8/255

Run the following command.
```bash
python train.py --config ./configs/cifar10_linf_model.yaml
```
We provide pre-trained checkpoints for adversarially trained model and preemptively robust model.

- ```adv_l2```: $\ell_2$ adversarially trained model with early stopping
- ```adv_linf```: $\ell_\infty$ adversarially trained model with early stopping
- ```preempt_robust_l2```: $\ell_2$ preemptively robust model
- ```preempt_robust_linf```: $\ell_\infty$ preemptively robust model

We also provide a pre-trained checkpoint for model with randomized smoothing.
- ```gaussian_0.1```: model trained with additive Gaussian noises ($\sigma=0.1$)

Shell scripts for downloading the models are located in ```./checkpoints/cifar10/wideresent/[train_type]/download.sh```. You can run each script to download a checkpoint named ```ckpt.pt```. To download all the checkpoints, run ```download_all_models.sh```.

## Generating preemptively robustified images and and their reconstructions

### 1. $\ell_2$ threat model, $\epsilon = \delta = 0.5$

Run the following command.
```bash
python robustify.py --config ./configs/cifar10_l2.yaml
python reconstruct.py --config ./configs/cifar10_l2.yaml
```

### 2. $\ell_\infty$ threat model, $\epsilon = \delta = 8/255$

Run the following command.
```bash
python robustify.py --config ./configs/cifar10_linf.yaml
python reconstruct.py --config ./configs/cifar10_linf.yaml
```

### 3. $\ell_2$ threat model, smoothed network, $\epsilon = \delta = 0.5$

Run the following command.
```bash
python robustify.py --config ./configs/cifar10_l2_rand.yaml
python reconstruct.py --config ./configs/cifar10_l2_rand.yaml
```
You can specify the classifier used for generating preemptively robust images by changing `train_type` in each yaml file.

## Grey-box attacks on preemptively robustified images

### 1. $\ell_2$ threat model, $\epsilon = \delta = 0.5$

Run the following command.
```bash
python attack_grey_box.py --config ./configs/cifar10_l2.yaml
```

### 2. $\ell_\infty$ threat model, $\epsilon = \delta = 8/255$

Run the following command.
```bash
python attack_grey_box.py --config ./configs/cifar10_linf.yaml
```

### 3. $\ell_2$ threat model, smoothed network, $\epsilon = \delta = 0.5$

Run the following command.
```bash
python attack_grey_box.py --config ./configs/cifar10_l2_rand.yaml
```

You can specify the attack setting by changing the Attack (eval) section in each yaml file. The default attack setting is 20-step PGD.

## White-box attacks on preemptively robustified images  

### 1. $\ell_2$ threat model, $\epsilon = \delta = 0.5$

Run the following command.
```bash
python attack_white_box.py --config ./configs/cifar10_l2.yaml
```

### 2. $\ell_\infty$ threat model, $\epsilon = \delta = 8/255$

Run the following command.
```bash
python attack_white_box.py --config ./configs/cifar10_linf.yaml
```  

### 3. $\ell_2$ threat model, smoothed network, $\epsilon = \delta = 0.5$

Run the following command.
```bash
python attack_white_box.py --config ./configs/cifar10_l2_rand.yaml
```

You can specify the attack setting by changing ```Attack (eval)``` section in each yaml file. The default attack setting is 20-step PGD.
