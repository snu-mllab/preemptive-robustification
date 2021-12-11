import sys
import time
import torch
import torch.nn as nn

from attacks import *


class PreemptRobustL2(object):
    def __init__(self, model,
                 attack_type, epsilon, step_size, num_steps, random_starts,
                 delta, lr, hessian=False, **kwargs):
        # Network
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()

        # Attack
        attack_class = getattr(sys.modules[__name__], attack_type)
        self.attack = attack_class(model, epsilon, step_size, num_steps)
        self.num_steps = num_steps
        self.random_starts = random_starts

        # Preemptive robustification
        self.delta = delta
        self.lr = lr
        self.hessian = hessian
    
    def random_perturb(self, x):
        noise = (torch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.delta)
        return torch.clamp(x + noise, 0, 1)

    def initialize(self, x, y, rec=False):
        self.x = x
        self.y = y
        self.x_rob = self.random_perturb(x) if not rec else x.clone() 
        self.rec = rec
        self.iter = 1

    def update(self):
        start = time.time()

        # Compute the distance
        dist = torch.mean(torch.norm((self.x_rob - self.x).view(self.x.shape[0], -1), dim=1))

        # Run attack
        x = self.x_rob.clone().detach().requires_grad_(True)
        y = self.y.clone().detach()

        total_grad = 0
        total_loss = 0

        for i in range(max(self.random_starts, 1)):
            x_adv = self.attack(
                x, y,
                random_start=(self.random_starts > 0),
                create_graph=self.hessian
            )
            
            # Compute loss and grad 
            output = self.model(x_adv)
            loss = self.criterion(output, y)
            grad, = torch.autograd.grad(torch.sum(loss), x)
    
            total_loss += torch.mean(loss)
            total_grad += grad

        total_loss /= max(self.random_starts, 1)
        total_grad /= max(self.random_starts, 1)
        
        direction = 1 if self.rec else -1

        # Update preemptively robustified image
        self.x_rob = self.x_rob + direction * self.lr * total_grad
        self.x_rob = self.project(self.x, self.x_rob)

        end = time.time()

        # Print
        print('Iter: {}, Loss: {:.4f}, Dist: {:.4f}, Time: {:.4f}'.format(
            self.iter, total_loss.item(), dist.item(), end - start))
        
        self.iter += 1
    
    def get_robust_img(self):
        return self.x_rob

    def project(self, x_orig, x):
        diff = x - x_orig
        diff = diff.renorm(p=2, dim=0, maxnorm=self.delta)
        x_proj = torch.clamp(x_orig + diff, 0, 1)
        return x_proj
        