from attacks import *
import sys
import time
import torch
import torch.nn as nn


class PreemptRobustL2Rand(object):
    def __init__(self, model,
                 attack_type, epsilon, step_size, num_steps, random_starts,
                 delta, lr, scale=0.1, num_samples=5, **kwargs):
        # Network
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()
        
        # Attack
        attack_class = getattr(sys.modules[__name__], attack_type)
        self.attack = attack_class(model, epsilon, step_size, num_steps, scale, num_samples)
        self.dynamics = self.attack.dynamics
        self.num_steps = num_steps
        self.random_starts = random_starts
        self.scale = scale
        self.num_samples = num_samples

        # Preemptive robustification
        self.delta = delta
        self.lr = lr
    
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
            )
            
            x_adv_noise = x_adv.unsqueeze(1).repeat(1, self.num_samples, 1, 1, 1).view(
                x_adv.shape[0]*self.num_samples, x_adv.shape[1], x_adv.shape[2], x_adv.shape[3])
            x_adv_noise = x_adv_noise + self.scale * torch.randn_like(x_adv_noise)
            output = self.model(x_adv_noise)
            prob = self.softmax(output)
            prob = prob.view(x_adv.shape[0], self.num_samples, -1)
            prob = torch.mean(prob, dim=1)
            one_hot = nn.functional.one_hot(y, num_classes=prob.shape[1])
            loss = -torch.log(torch.sum(prob*one_hot, dim=1))
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
