import torch
import torch.nn as nn


class Dynamics(nn.Module):
    def __init__(
        self, 
        model, 
        epsilon, 
        step_size
    ):
        super(Dynamics, self).__init__()

        # Network
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='sum').cuda()

        # Attack
        self.epsilon = epsilon
        self.step_size = step_size

    def forward(
        self, 
        x_orig, 
        x, 
        y, 
        create_graph=False, 
        targeted=False
    ):
        # Compute the loss and the gradient
        # If you want to add the gradient into the computational graph, set create_graph = True
        output = self.model(x)
        loss = self.criterion(output, y)
        grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # Update the image
        x_next = self.step(x, -grad if targeted else grad)
        x_proj = self.project(x_orig, x_next)

        return x_proj

    def step(self, x, g):
        raise NotImplementedError

    def project(self, x_orig, x):
        raise NotImplementedError

    def random_perturb(self, x):
        raise NotImplementedError


class L2Dynamics(Dynamics):
    def step(self, x, g):
        g_norm = torch.norm(g.view(g.shape[0], -1), 
                            dim=1).view(-1, *([1] * (len(x.shape) - 1)))
        scaled_g = g / (g_norm + 1e-10)
        x_next = x + self.step_size * scaled_g
        return x_next

    def project(self, x_orig, x):
        diff = x - x_orig
        diff = diff.renorm(p=2, dim=0, maxnorm=self.epsilon)
        x_proj = torch.clamp(x_orig + diff, 0, 1)
        return x_proj

    def random_perturb(self, x):
        noise = (torch.rand_like(x) - 0.5).renorm(p=2, dim=0, 
                                                  maxnorm=self.epsilon)
        return torch.clamp(x + noise, 0, 1)


class LinfDynamics(Dynamics):
    def step(self, x, g):
        x_next = x + self.step_size * torch.sign(g)
        return x_next

    def project(self, x_orig, x):
        diff = x - x_orig
        diff = torch.clamp(diff, -self.epsilon, self.epsilon)
        x_proj = torch.clamp(x_orig + diff, 0, 1)
        return x_proj

    def random_perturb(self, x):
        noise = 2 * (torch.rand_like(x) - 0.5) * self.epsilon
        return torch.clamp(x + noise, 0, 1)
