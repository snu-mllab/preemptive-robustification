from attacks import *
import sys
import time
import torch
import torch.nn as nn


class PreemptRobustLinf(object):
    def __init__(self, model,
                 attack_type, epsilon, step_size, num_steps, random_starts,
                 delta, lr, **kwargs):
        # Network
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()

        # Attack
        attack_class = getattr(sys.modules[__name__], attack_type)
        self.attack = attack_class(model, epsilon, step_size, num_steps)
        self.num_steps = num_steps
        self.random_starts = random_starts

        # Preemptively robustification
        self.delta = delta
        self.lr = lr
    
    def random_perturb(self, x):
        noise = 2 * (torch.rand_like(x) - 0.5) * self.delta
        return torch.clamp(x + noise, 0, 1)

    def initialize(self, x, y, rec=False):
        self.x = x
        self.y = y
        x_init = self.random_perturb(x) if not rec else x.clone()
        self.x_rob = nn.Parameter(self.transform(x, x_init))
        self.rec = rec
        self.iter = 1
        self.optimizer = torch.optim.RMSprop([self.x_rob], lr=self.lr)
    
    def update(self):
        start = time.time()

        # Compute the distance
        dist = torch.mean(torch.amax(torch.abs(self.transform_inv(self.x, self.x_rob) - self.x).view(self.x.size(0), -1), dim=1))

        # Run attack
        x_t = self.x_rob.clone().detach().requires_grad_(True)
        x = self.transform_inv(self.x, x_t)
        y = self.y.clone().detach()

        total_grad = 0
        total_loss = 0

        for i in range(max(self.random_starts, 1)):
            x_adv = self.attack(
                x, y,
                random_start=(self.random_starts > 0)
            )

            # Compute loss and grad
            output = self.model(x_adv)
            loss = self.criterion(output, y)
            grad, = torch.autograd.grad(torch.sum(loss), x_t)
           
            total_loss += torch.mean(loss)
            total_grad += grad

        total_loss /= max(self.random_starts, 1)
        total_grad /= max(self.random_starts, 1)
        
        direction = 1 if not self.rec else -1

        # Update the safe spot
        self.optimizer.zero_grad()
        self.x_rob.grad = direction * total_grad
        self.optimizer.step()

        end = time.time()

        # Print
        print('iter: {}, loss: {:.4f}, dist: {:.4f}, time: {:.4f}'.format(
            self.iter, total_loss.item(), dist.item(), end - start))
        
        self.iter += 1

    def get_robust_img(self):
        return self.transform_inv(self.x, self.x_rob.data)

    # [0, 1]^d -> R^d
    def transform(self, x_orig, x):
        lower = torch.clamp(x_orig - self.delta, min=0)
        upper = torch.clamp(x_orig + self.delta, max=1)
        
        def atanh(t):
            return 0.5 * torch.log((1 + t) / (1 - t))

        return atanh(((x - lower) / (upper - lower) - 0.5) * 1.999)

    # R^d -> [0, 1]^d
    def transform_inv(self, x_orig, x):
        lower = torch.clamp(x_orig - self.delta, min=0)
        upper = torch.clamp(x_orig + self.delta, max=1)
        return lower + (torch.tanh(x) + 1) / 2 * (upper - lower)
