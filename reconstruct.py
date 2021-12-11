import argparse
import math
import numpy as np
import os
import random
import sys
import yaml

import torch

from defenses import *
from utils.eval_utils import evaluate_rand
from utils.model_utils import create_model
from utils.logging_utils import print_params 


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
    total_inputs_rob = np.load(
        os.path.join(output_dir, 'inputs_rob_{}_{}.npy'.format(start, end))
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
    
    # Create defense
    defense_class = getattr(sys.modules[__name__], args['defense_type'])
    if not args['rand']:
        defense = defense_class(
            model=model, 
            attack_type=args['attack_type'], 
            epsilon=args['epsilon'],
            step_size=args['step_size'], 
            num_steps=args['num_steps'],
            random_starts=args['random_starts'],
            delta=args['delta'],
            lr=args['lr'],
            hessian=args['hessian']
        )
    else:
        defense = defense_class(
            model=model, 
            attack_type=args['attack_type'], 
            epsilon=args['epsilon'], 
            step_size=args['step_size'], 
            num_steps=args['num_steps'],
            random_starts=args['random_starts'],
            delta=args['delta'],
            lr=args['lr'],
            hessian=args['hessian'],
            scale=args['scale'],
            num_samples=args['num_samples']
        )

    # Reconstruct original image
    num_images = len(total_inputs_rob)
    num_batches = int(math.ceil(num_images / args['batch_size']))
    
    total_inputs_rec = None

    for batch_idx in range(num_batches): 
        bstart = batch_idx * args['batch_size']
        bend = min(bstart + args['batch_size'], num_images)

        print('\nImage {}-{}'.format(args['start'] + bstart, args['start'] + bend))

        inputs_rob = total_inputs_rob[bstart:bend, ...]        
        inputs_rob = torch.from_numpy(inputs_rob).float()
        inputs_rob = inputs_rob.cuda()
        
        print('\nReconstructing original images') 
        
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

        defense.initialize(inputs_rob, preds_rob, rec=True)
        for i in range(args['num_iters']):
            defense.update()
        inputs_rec = defense.get_robust_img()
        inputs_rec = inputs_rec.detach().cpu().numpy()

        total_inputs_rec = inputs_rec if total_inputs_rec is None \
            else np.concatenate([total_inputs_rec, inputs_rec], axis=0)
   
    print('\nSaving')
    
    np.save(
        os.path.join(output_dir, 'inputs_rec_{}_{}.npy'.format(
            args['start'], args['start'] + args['num_images'])), 
        total_inputs_rec
    )
