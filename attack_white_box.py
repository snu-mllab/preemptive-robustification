import argparse
import math
import numpy as np
import os
import random
import sys
import yaml

import torch

from attacks import *
from utils.eval_utils import evaluate_rand
from utils.model_utils import create_model
from utils.logging_utils import print_params, AverageMeter 


# Arguments
parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--ckpt_dir', default='./checkpoints', type=str)
parser.add_argument('--config', default='./configs/cifar10_l2.yaml', type=str)

args = parser.parse_args()

opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))
args = opt


# Main script
if __name__ == '__main__':
    # Fix random seed
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    # Print arguments
    print_params(args) 
    
    # Create directory
    output_dir = os.path.join(
        args['output_dir'], 
        args['data_type'].lower(), 
        args['train_type'], 
        args['defense_type'].lower()
    )

    # Load dataset
    print('\nLoading dataset')
    start = args['start']
    end = args['start'] + args['num_images']
    
    total_inputs = np.load(
        os.path.join(output_dir, 'inputs_{}_{}.npy'.format(start, end))
    ) 
    total_inputs_rob = np.load(
        os.path.join(output_dir, 'inputs_rob_{}_{}.npy'.format(start, end))
    ) 
    total_inputs_rec = np.load(
        os.path.join(output_dir, 'inputs_rec_{}_{}.npy'.format(start, end))
    ) 
    total_targets = np.load(
        os.path.join(output_dir, 'targets_{}_{}.npy'.format(start, end))
    ) 

    # Create model
    print('\nCreating model')
    ckpt_path = os.path.join(
        args['ckpt_dir'], 
        args['data_type'].lower(), 
        args['model_type'].lower(), 
        args['train_type'], 
        'ckpt.pt'
    )
    model = create_model(args['data_type'], args['model_type'], ckpt_path)
    
    # Create attack
    attack_class = getattr(sys.modules[__name__], args['attack_type_eval'])
    if not args['rand']:
        attack = attack_class(
            model, 
            epsilon=args['wbox_epsilon_p'], 
            step_size=5*args['wbox_epsilon_p']/args['num_steps_eval'], 
            num_steps=args['num_steps_eval'],
            wbox_epsilon=args['wbox_epsilon']    
        )
    else:
        attack = attack_class(
            model, 
            epsilon=args['wbox_epsilon_p'], 
            step_size=5*args['wbox_epsilon_p']/args['num_steps_eval'], 
            num_steps=args['num_steps_eval'],
            scale=args['scale'],
            num_samples=args['num_samples_eval'],
            wbox_epsilon=args['wbox_epsilon']
        )
    
    # Logger
    acc_meter = AverageMeter('acc', ':.4f')
    acc_adv_meter = AverageMeter('acc_adv', ':.4f')

    # Eval
    num_images = len(total_inputs_rob)
    num_batches = int(math.ceil(num_images / args['batch_size_eval']))
    
    total_corrects = None
    total_corrects_adv = None

    for batch_idx in range(num_batches): 
        bstart = batch_idx * args['batch_size_eval']
        bend = min(bstart + args['batch_size_eval'], num_images)

        inputs = total_inputs[bstart:bend, ...]
        inputs_rob = total_inputs_rob[bstart:bend, ...]
        inputs_rec = total_inputs_rec[bstart:bend, ...]
        targets = total_targets[bstart:bend, ...]
        
        inputs = torch.from_numpy(inputs).float()
        inputs_rob = torch.from_numpy(inputs_rob).float()
        inputs_rec = torch.from_numpy(inputs_rec).float()
        targets = torch.from_numpy(targets).long() 
        
        inputs = inputs.cuda()
        inputs_rob = inputs_rob.cuda()
        inputs_rec = inputs_rec.cuda()
        targets = targets.cuda()

        if not args['rand']:
            with torch.no_grad():
                outputs_rob = model(inputs_rob)
            preds_rob = torch.max(outputs_rob, dim=1)[1]
        else:
            preds_rob = evaluate_rand(
                model, 
                inputs_rob, 
                scale=args['scale'], 
                num_samples_test=50
            )
             
        corrects = (preds_rob == targets).long()
        num_corrects = corrects.sum().item()
        acc = num_corrects / targets.size(0)
        acc_meter.update(acc, targets.size(0))
        
        # Run attack
        inputs_adv = attack(inputs_rec, targets, wbox=True, x_o=inputs)

        if not args['rand']:
            with torch.no_grad():
                outputs_adv = model(inputs_adv)
            preds_adv = torch.max(outputs_adv, dim=1)[1]
        else:
            preds_adv = evaluate_rand(
                model, 
                inputs_adv, 
                scale=args['scale'], 
                num_samples_test=50
            )

        if args['attack_type_eval'].endswith('L2'):
            dists = torch.norm(
                (inputs_adv - inputs).view(inputs.size(0), -1), 
                dim=1
            ) 
        elif args['attack_type_eval'].endswith('Linf'): 
            dists = torch.amax(
                torch.abs((inputs_adv - inputs).view(inputs.size(0), -1)), 
                dim=1
            )
        else:
            dists = None

        corrects_adv = torch.logical_or(
            preds_adv == targets, 
            dists > args['wbox_epsilon']
        ).long()
        corrects_adv = torch.logical_and(corrects, corrects_adv) 
        num_corrects_adv = corrects_adv.sum().item()
        acc_adv = num_corrects_adv / targets.size(0)
        acc_adv_meter.update(acc_adv, targets.size(0))

        corrects = corrects.detach().cpu().numpy()
        corrects_adv = corrects_adv.detach().cpu().numpy()

        total_corrects = corrects if total_corrects is None \
            else np.concatenate([total_corrects, corrects], axis=0)
        total_corrects_adv = corrects_adv if total_corrects_adv is None \
            else np.concatenate([total_corrects_adv, corrects_adv], axis=0)
        
        print('[{}/{}] acc: {:.2f}%, acc_adv: {:.2f}%'.format(
            batch_idx + 1, num_batches, acc_meter.avg * 100, acc_adv_meter.avg * 100
        ))
