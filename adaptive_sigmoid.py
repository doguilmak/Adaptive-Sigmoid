import torch
import torch.nn as nn

class AdaptiveSigmoidLayer(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha should be positive to avoid instability.")
        self.alpha = alpha

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * x))