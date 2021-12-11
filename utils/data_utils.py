import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as transforms


SIZE_CIFAR10 = 32
TRANSFORM_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
])

SIZE_IMAGENET = 224
TRANSFORM_IMAGENET = transforms.Compose([
    transforms.Resize(int(SIZE_IMAGENET*1.14)),
    transforms.CenterCrop(SIZE_IMAGENET),
    transforms.ToTensor(),
])


def load_dataset(data_type, data_dir, batch_size):
    if data_type == 'CIFAR10':
        dataset = CIFAR10(root=data_dir, train=False, download=False, transform=TRANSFORM_CIFAR10)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    elif data_type == 'ImageNet':
        dataset = ImageFolder(os.path.join(data_dir, 'ImageNet/test'), transform=TRANSFORM_IMAGENET)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    else:
        raise NotImplementedError 

