import torch.nn as nn

# model wrapper to normalize images before actual model pass
class ModelWrapper(nn.Module):
  def __init__(self, model, mean, std):
    super(ModelWrapper, self).__init__()
    self.model = model
    self.mean = mean
    self.std = std

  def forward(self, x):
    return self.model(self.normalize(x))

  def normalize(self, x):
    return (x - self.mean[:, None, None]) / self.std[:, None, None]

