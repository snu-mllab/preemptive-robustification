import torch
import torch.nn as nn


class DynamicsRand(nn.Module):
    def __init__(
        self, 
        model, 
        epsilon, 
        step_size, 
        scale, 
        num_samples
    ):
        super(DynamicsRand, self).__init__()

        # Network
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

        # Attack
        self.epsilon = epsilon
        self.step_size = step_size
        self.scale = scale
        self.num_samples = num_samples

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
        x_noise = x.unsqueeze(1).repeat(1, self.num_samples, 1, 1, 1).view(
            x.shape[0]*self.num_samples, x.shape[1], x.shape[2], x.shape[3]) 
        x_noise = x_noise + self.scale * torch.randn_like(x_noise)
        output = self.model(x_noise)
        prob = self.softmax(output)
        prob = prob.view(x.shape[0], self.num_samples, -1)
        prob = torch.mean(prob, dim=1) 
        one_hot = nn.functional.one_hot(y, num_classes=prob.shape[1])  
        loss = -torch.log(torch.sum(prob*one_hot, dim=1))
        grad, = torch.autograd.grad(torch.mean(loss), x, create_graph=create_graph)

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


class L2DynamicsRand(DynamicsRand):
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
