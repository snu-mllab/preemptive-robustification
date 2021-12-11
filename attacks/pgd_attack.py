from .dynamics import *
from .dynamics_rand import *
import torch.nn as nn


class PGDAttack(nn.Module):
    def __init__(self, model, epsilon, step_size, num_steps):
        super(PGDAttack, self).__init__()

        # Network
        self.model = model

        # Attack
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps

    def forward(self, x, y, random_start=True, create_graph=False, targeted=False, detach=False, **kwargs):
        # Turn on the gradient computation
        x.requires_grad_(True)

        # Add a random noise to the image
        if random_start:
            x_adv = self.dynamics.random_perturb(x)
        else:
            x_adv = x.clone()

        # Attack the image
        for _ in range(self.num_steps):
            x_adv = self.dynamics(x, x_adv, y, create_graph=create_graph, targeted=targeted)
            if detach:
                x_adv = x_adv.clone().detach().requires_grad_(True)
        
        if detach:
            x_adv.requires_grad_(False)

        return x_adv

# l2 threat
class PGDAttackL2(PGDAttack):
    def __init__(self, model, epsilon=0.5, step_size=0.125, num_steps=20, **kwargs):
        super(PGDAttackL2, self).__init__(model, epsilon, step_size, num_steps)
        # Dynamics
        self.dynamics = L2Dynamics(model, epsilon, step_size)

# l2 threat, randomized smoothing
class PGDAttackL2Rand(PGDAttack):
    def __init__(self, model, epsilon=0.5, step_size=0.125, num_steps=20, 
                 scale=0.1, num_samples=5, **kwargs):
        super(PGDAttackL2Rand, self).__init__(model, epsilon, step_size, num_steps)
        # Dynamics
        self.dynamics = L2DynamicsRand(model, epsilon, step_size, scale, num_samples)

# linf threat
class PGDAttackLinf(PGDAttack):
    def __init__(self, model, epsilon=8/255, step_size=2/255, num_steps=20, **kwargs):
        super(PGDAttackLinf, self).__init__(model, epsilon, step_size, num_steps)
        # Dynamics
        self.dynamics = LinfDynamics(model, epsilon, step_size)
