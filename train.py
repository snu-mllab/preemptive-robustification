import argparse
import numpy as np
import os
import random
import sys
import time
import yaml

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from attacks import *
from models import *
from utils.logging_utils import print_params, AverageMeter, ProgressMeter

# Argument
parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--data_dir', default='./data', type=str)
parser.add_argument('--ckpt_dir', default='./checkpoints')

# Experiment
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--save_every', default=10, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--config', default='./configs/cifar10_l2_model.yaml', type=str)

# Data
parser.add_argument('--data_type', default='CIFAR10', type=str)

# Model
parser.add_argument('--model_type', default='WideResNet', type=str)

# Training
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--max_epoch', default=200, type=int)
parser.add_argument('--network_lr', default=1e-1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

args = parser.parse_args()

opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))
args = opt


# Transform (train)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# Transform (test)
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Pixel statistics
mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()


# Main script
if __name__ == '__main__':
    # Fix seed
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed']) 
    random.seed(args['seed'])

    # Print arguments
    print_params(args)

    # Directory
    ckpt_dir = os.path.join(
        args['ckpt_dir'], 
        args['data_type'].lower(), 
        args['model_type'].lower(), 
        args['train_type'],
    ) 
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dataset
    print('\nLoading data')
    train_set = torchvision.datasets.CIFAR10(
        root=args['data_dir'], 
        train=True, 
        download=False, 
        transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args['batch_size'], 
        shuffle=True,
        num_workers=4
    )

    test_set = torchvision.datasets.CIFAR10(
        root=args['data_dir'], 
        train=False, 
        download=False, 
        transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=100, 
        shuffle=False, 
        num_workers=4
    )

    # Create model
    print('\nCreating model')
    model_class = getattr(sys.modules[__name__], args['model_type'])    
    base_model = model_class()
    model = base_model.cuda()
    model = torch.nn.DataParallel(model)

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args['network_lr'],
        momentum=0.9,
        weight_decay=args['weight_decay']
    )
    scheduler= torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[int(0.5*args['max_epoch']), int(0.75*args['max_epoch'])], 
        gamma=0.1
    )

    # Create attack
    attack_class = getattr(sys.modules[__name__], args['attack_type'])
    attack = attack_class(model, args['epsilon'], args['step_size'], args['num_steps'])

    # Create defense
    defense_class = getattr(sys.modules[__name__], args['defense_type'])
    defense = defense_class(model, args['delta'], args['lr'], args['num_iters'])

    # Load checkpoint
    start_epoch = 0

    def train(epoch):
        print('\nEpoch: {}'.format(epoch))
        print('\nTraining')
    
        batch_time_meter = AverageMeter('time', ':.2f')
        lr_meter = AverageMeter('lr', ':.6f')
        xent_meter = AverageMeter('xent', ':.3f')
        acc_meter = AverageMeter('acc', ':.3f')
        xent_adv_meter = AverageMeter('xent_adv', ':.3f')
        acc_adv_meter = AverageMeter('acc_adv', ':.3f')

        progress = ProgressMeter(
            len(train_loader),
            [batch_time_meter, lr_meter, xent_meter, acc_meter, xent_adv_meter, acc_adv_meter],
            prefix="Epoch: [{}]".format(epoch)
        )
        
        model.train()
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # Run defense
            model.eval()
            inputs_rob = defense(inputs, targets, detach=False)

            # Run attack
            model.eval()
            inputs_adv = attack(inputs_rob, targets, detach=False)
          
            # Run forward pass
            model.train()
            outputs = model(inputs)
            outputs_adv = model(inputs_adv)

            # Compute loss
            xent = F.cross_entropy(outputs, targets) 
            xent_adv = F.cross_entropy(outputs_adv, targets)
            loss = xent_adv
        
            # Run backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            lr = optimizer.param_groups[0]['lr']
            preds = outputs.max(1)[1]
            preds_adv = outputs_adv.max(1)[1]
            correct = preds.eq(targets).sum().item()
            correct_adv = preds_adv.eq(targets).sum().item()
            acc = correct / targets.size(0)
            acc_adv = correct_adv / targets.size(0)
            
            lr_meter.update(lr)
            xent_meter.update(xent.item(), targets.size(0))
            acc_meter.update(acc, targets.size(0))
            xent_adv_meter.update(xent_adv.item(), targets.size(0))
            acc_adv_meter.update(acc_adv, targets.size(0))
            batch_time_meter.update(time.time() - end)
            
            end = time.time()

            # Logging
            if batch_idx % args['print_every'] == 0:
                progress.display(batch_idx)

        scheduler.step()
   
    def test(epoch):
        print('\nTesting...')
        
        batch_time_meter = AverageMeter('time', ':.2f')
        xent_meter = AverageMeter('xent', ':.3f')
        acc_meter = AverageMeter('acc', ':.3f')
        xent_adv_meter = AverageMeter('xent_adv', ':.3f')
        acc_adv_meter = AverageMeter('acc_adv', ':.3f')
        
        model.eval() 
        end = time.time()

        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # Run attack
            inputs_adv = attack(inputs, targets)
        
            # Run forward pass   
            outputs = model(inputs)
            outputs_adv = model(inputs_adv)

            # Compute loss
            xent = F.cross_entropy(outputs, targets)
            xent_adv = F.cross_entropy(outputs_adv, targets)

            # Statistics
            preds = outputs.max(1)[1]
            preds_adv = outputs_adv.max(1)[1] 
            correct = preds.eq(targets).sum().item()
            correct_adv = preds_adv.eq(targets).sum().item()
            acc = correct / targets.size(0)
            acc_adv = correct_adv / targets.size(0)
            
            xent_meter.update(xent.item(), targets.size(0))
            acc_meter.update(acc, targets.size(0))
            xent_adv_meter.update(xent_adv.item(), targets.size(0))
            acc_adv_meter.update(acc_adv, targets.size(0))
            batch_time_meter.update(time.time() - end)
            
            end = time.time()
        
        # Logging
        print('xent {:.3f}, acc: {:.3f}, xent_adv: {:.3f}, acc_adv: {:.3f}'.format(
            xent_meter.avg, acc_meter.avg, xent_adv_meter.avg, acc_adv_meter.avg)
        )
            
        # Save checkpoint
        if epoch % args['save_every'] == 0:
            print('\nSaving')
            state_dict = {
                'epoch': epoch,
                'state_dict': base_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state_dict, os.path.join(ckpt_dir, 'ckpt.pt'))
    
    for epoch in range(start_epoch + 1, args['max_epoch'] + 1):
        train(epoch)
        test(epoch)
