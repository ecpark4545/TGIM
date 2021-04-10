import torch
import torch.nn as nn
import torch.nn.functional as F
from .switchable_norm import SwitchNorm2d


class ResidualBlock_TaGAN(nn.Module):
  def __init__(self, ndim):
    super(ResidualBlock_TaGAN, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
        nn.BatchNorm2d(ndim),
        nn.ReLU(inplace=True),
        nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
        nn.BatchNorm2d(ndim),
    )

  def forward(self, x):
    return x + self.encoder(x)


class ResidualBlock_RelGAN(nn.Module):
  def __init__(self, ndim):
    super(ResidualBlock_RelGAN, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(ndim, ndim, 3, padding=1, stride=1),
        SwitchNorm2d(n_out, momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(ndim, ndim, 3, padding=1, stride=1),
        SwitchNorm2d(n_out, momentum=0.9),
    )
  def forward(self, x):
    return x + self.encoder(x)



