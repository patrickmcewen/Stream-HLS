from typing import Callable, Optional
from torch import nn, Tensor
import torch.nn.functional as F
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

def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)

        # STE using detach
        x_quant = x_norm #+ (activation_quant(x_norm) - x_norm).detach()
        w_quant = w #+ (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y



def default(val, d):
    return val if val is not None else d


def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)


# [GLU]
class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
        linear (Callable, optional): Linear function to be used for projection. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable,
        mult_bias: bool = False,
        linear: Callable = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias

        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear(dim_in, dim_out * 4, *args, **kwargs)

        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate) * self.mult_bias


# [FEATURE] Add type hints to the forward method
class BitFeedForward(nn.Module):
    """
    BitFeedForward module performs feed-forward operations on the input tensor.

    Args:
        dim (int): The input dimension.
        dim_out (int, optional): The output dimension. If not provided, it is set to the input dimension.
        mult (int, optional): The multiplier for the inner dimension. Default is 4.
        glu (bool, optional): Whether to use Gated Linear Unit (GLU) activation. Default is False.
        glu_mult_bias (bool, optional): Whether to apply bias to the GLU activation. Default is False.
        swish (bool, optional): Whether to use Swish activation. Default is False.
        relu_squared (bool, optional): Whether to use squared ReLU activation. Default is False.
        post_act_ln (bool, optional): Whether to apply Layer Normalization after activation. Default is False.
        dropout (float, optional): The dropout probability. Default is 0.0.
        no_bias (bool, optional): Whether to exclude bias in linear layers. Default is False.
        zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Default is False.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        glu_mult_bias: bool = False,
        swish: bool = False,
        post_act_ln: bool = False,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
        else:
            project_in = nn.Sequential(
                BitLinear(dim, inner_dim, bias=not no_bias, *args, **kwargs), activation
            )
        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                BitLinear(inner_dim, dim_out, bias=not no_bias, *args, **kwargs),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                BitLinear(inner_dim, dim_out, bias=not no_bias, *args, **kwargs),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the BitFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.ff(x)