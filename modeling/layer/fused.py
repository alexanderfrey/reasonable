# fused.py
import torch
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union, Dict


class FusedLinear(nn.Module):
    """
    Memory-efficient linear layer with fused operations for linear transform,
    activation, and dropout.
    """

    SUPPORTED_ACTIVATIONS = {
        "gelu": F.gelu,
        "relu": F.relu,
        "silu": F.silu,
        None: lambda x: x
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        activation: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize FusedLinear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias term
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', 'silu', or None)
            device: Torch device
            dtype: Tensor dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # Validate inputs
        if not (0 <= dropout < 1.0):
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}. "
                           f"Choose from {list(self.SUPPORTED_ACTIVATIONS.keys())}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation_fn = self.SUPPORTED_ACTIVATIONS[activation]

        # Initialize parameters with proper dtype/device
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(activation)

    def reset_parameters(self, activation: Optional[str] = None) -> None:
        """
        Reset layer parameters with activation-aware initialization.
        """
        if activation == 'gelu':
            # Use He initialization adjusted for GELU
            gain = 1.0 / math.sqrt(0.8862)
            nn.init.kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
            self.weight.data *= gain
        elif activation == 'relu':
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        else:
            # Default initialization for linear/other activations
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused operations.
        
        Args:
            input: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Validate input shape
        if input.size(-1) != self.in_features:
            raise ValueError(f"Expected input features: {self.in_features}, "
                           f"got: {input.size(-1)}")

        if torch.jit.is_scripting() or not self.training:
            # Standard forward pass when scripting or in eval mode
            output = F.linear(input, self.weight, self.bias)
            output = self.activation_fn(output)
            if self.dropout > 0 and self.training:
                output = F.dropout(output, p=self.dropout, training=True)
            return output
        else:
            # Fused operations for training mode
            if hasattr(torch.nn.functional, 'fused_linear'):
                # Use fused kernel if available
                return F.fused_linear(
                    input, 
                    self.weight,
                    self.bias,
                    self.activation_fn,
                    self.dropout if self.training else 0.0
                )
            else:
                # Fallback to standard operations
                output = F.linear(input, self.weight, self.bias)
                output = self.activation_fn(output)
                if self.dropout > 0:
                    output = F.dropout(output, p=self.dropout, training=True)
                return output

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'dropout={self.dropout}, '
                f'activation={[k for k, v in self.SUPPORTED_ACTIVATIONS.items() '
                f'if v == self.activation_fn][0]}')