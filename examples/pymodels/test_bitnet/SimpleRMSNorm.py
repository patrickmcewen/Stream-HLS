from torch import nn, Tensor
import torch

class SimpleRMSNorm(nn.Module):
    """
    Simple RMS Normalization module.
    
    Args:
        dim (int): The dimension to normalize over.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
    
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        return x / rms * self.scale