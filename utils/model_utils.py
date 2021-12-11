import sys
import torch

from models import *


MEAN_CIFAR10 = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
STD_CIFAR10 = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()

MEAN_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).cuda()
STD_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).cuda()


def create_model(data_type, model_type, ckpt_path):
    model_class = getattr(sys.modules[__name__], model_type)
    base_model = model_class()
    model = base_model.cuda()
    model = torch.nn.DataParallel(model)
    
    if data_type == 'CIFAR10':
        model = ModelWrapper(model, MEAN_CIFAR10, STD_CIFAR10)
    elif data_type == 'ImageNet':
        model = ModelWrapper(model, MEAN_IMAGENET, STD_IMAGENET)
    else:
        raise NotImplementedError

    checkpoint = torch.load(ckpt_path)
    state_dict = {}
    state_dict_old = None

    if 'state_dict' in checkpoint:
        state_dict_old = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict_old = checkpoint['model']
    else:
        state_dict_old = checkpoint

    for k, v in state_dict_old.items():
        if k.startswith('1.'):
            k = k[len('1.'):]
        if k.startswith('model.'):
            k = k[len('model.'):]
        if k.startswith('module.'):
            k = k[len('module.'):]
        if k.startswith('attacker.'):
            k = k[len('attacker.'):]
        if k.startswith('model.'):
            k = k[len('model.'):]
        if k.startswith('normalize'):
            continue
        state_dict[k] = v        
    
    base_model.load_state_dict(state_dict)

    model.eval()
    
    return model

