import argparse
import numpy as np
import os
import random
import sys
import yaml

import torch

from defenses import *
from utils.data_utils import load_dataset
from utils.eval_utils import evaluate_rand
from utils.model_utils import create_model
from utils.logging_utils import print_params 


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', type=str)
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
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print('\nLoading dataset')
    loader = load_dataset(args['data_type'], args['data_dir'], args['batch_size'])
    
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
        defense = defense_class(model=model, attack_type=args['attack_type'], 
                                epsilon=args['epsilon'], step_size=args['step_size'], 
                                num_steps=args['num_steps'], random_starts=args['random_starts'],
                                delta=args['delta'], lr=args['lr'], hessian=args['hessian'])
    else:
        defense = defense_class(model=model, attack_type=args['attack_type'], 
                                epsilon=args['epsilon'], step_size=args['step_size'], 
                                num_steps=args['num_steps'], random_starts=args['random_starts'],
                                delta=args['delta'], lr=args['lr'], hessian=args['hessian'],
                                scale=args['scale'], num_samples=args['num_samples'])

    # Reconstruct original image
    count = 0
    total_inputs = None
    total_inputs_rob = None
    total_targets = None

    for inputs, targets in loader: 
        count += args['batch_size']

        if count <= args['start']:
            continue

        print('\nImage {}-{}'.format(count - args['batch_size'], count))
        
        # Infer labels
        inputs, targets = inputs.cuda(), targets.cuda()    
        if not args['rand']:
            with torch.no_grad():
                outputs = model(inputs)
            preds = torch.max(outputs, dim=1)[1]
        else:
            preds = evaluate_rand(model, inputs, scale=args['scale'], num_samples_test=50)

        # Generate preemptively robustified images
        print('\nGenerating preemptively robustified images')
        defense.initialize(inputs, preds, rec=False)
        for i in range(args['num_iters']):
            defense.update()
        inputs_rob = defense.get_robust_img()
      
        inputs = inputs.detach().cpu().numpy()
        inputs_rob = inputs_rob.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        total_inputs = inputs if total_inputs is None \
            else np.concatenate([total_inputs, inputs], axis=0)
        total_inputs_rob = inputs_rob if total_inputs_rob is None \
            else np.concatenate([total_inputs_rob, inputs_rob], axis=0)
        total_targets = targets if total_targets is None \
            else np.concatenate([total_targets, targets], axis=0)
        
        if count >= args['start'] + args['num_images']:
            break
   
    print('\nSaving')
    np.save(
        os.path.join(output_dir, 'inputs_{}_{}.npy'.format(
            args['start'], args['start'] + args['num_images'])), 
        total_inputs
    )
    np.save(
        os.path.join(output_dir, 'inputs_rob_{}_{}.npy'.format(
            args['start'], args['start'] + args['num_images'])), 
        total_inputs_rob
    )
    np.save(
        os.path.join(output_dir, 'targets_{}_{}.npy'.format(
            args['start'], args['start'] + args['num_images'])),
        total_targets
    )
