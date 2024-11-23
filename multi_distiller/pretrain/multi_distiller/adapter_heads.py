# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from itertools import chain
import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.functional import interpolate
from typing import Tuple, Union

class LSTMAdapterHead(nn.Module):
    def __init__(self, source_size: Union[Tuple[int, ...], torch.Size], target_model: str, hidden_size_factor: Union[int, float] = 1.0):
        super().__init__()

        self.hidden_size_factor = hidden_size_factor
        source_channel_size = source_size[-1]  # Feature dimension (C)
        hidden_dim = int(source_channel_size * hidden_size_factor)  # Adjusting based on the channel dimension
        self.source_size = source_size
        self.target_model = target_model
        self.target_length = 768

        # Define the LSTM adapter
        self.lstm = nn.LSTM(
            input_size=source_channel_size * source_channel_size,  # Channel size of input
            hidden_size=hidden_dim,  # Adjusted hidden size
            num_layers=2,  # LSTM depth
            batch_first=True,  # Input and output tensors are [B, T, H, C]
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, source_channel_size),  # Map back to the original channel size
            nn.LayerNorm(source_channel_size)
        )

        self.length_adjuster = nn.Sequential(
            nn.LayerNorm(source_channel_size),  # Apply LayerNorm over the last dimension
            # Adaptive pooling to set H to a constant size
            nn.AdaptiveAvgPool2d((self.target_length, source_channel_size)),  # Resize to target length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LSTMAdapterHead.

        Args:
            x (torch.Tensor): input tensor with shape [B, T, H, C].

        Returns:
            torch.Tensor: transformed tensor matching the input shape.
        """
        # Apply LSTM without reshaping
        x = self.length_adjuster(x)
        B, T, H, C = x.shape
        x = x.reshape(B, T, -1)
        lstm_out, _ = self.lstm(x)  # Output from LSTM: [B, T, hidden_dim]
        x = self.fc(lstm_out)  # Map back to the original feature size

        return x




class Interpolation(nn.Module):
    """Interpolation nn.Module wrap for nn.functional.interpolate.

    Attributes:
        target_size (Tuple[int, int] | torch.Size): target spatial size of this interpolation.
    """

    def __init__(self, target_size: Union[Tuple[int, int], torch.Size]) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Very simple forward pass to call interpolate()."""
        return interpolate(x, self.target_size)


class LightConvAdapterHead(nn.Module):
    """Light Convolutional Adapter module.

    Transforms features from source size [B, T, H, C] to target size [B, T, H, C].
    Uses CNN to map channel and spatial sizes jointly, while preserving the overall shape.
    """

    def __init__(
        self,
        source_size: Union[Tuple[int, ...], torch.Size],
        target_model: str,
        hidden_size_factor: Union[int, float] = 1.0,
    ):
        """Initialization function for LightConvAdapterHead.

        Args:
            source_size (Union[Tuple[int, ...], torch.Size]): the size of the source feature.
            hidden_size_factor (Union[int, float]): the size of hidden dim of feature translator
                as a factor of input feature hidden dim.
        """
        super().__init__()

        self.hidden_size_factor = hidden_size_factor
        source_channel_size = source_size[-1]  # Feature dimension (C)
        hidden_dim = int(source_channel_size * hidden_size_factor)  # Adjusting based on the channel dimension
        self.source_size = source_size
        self.target_length = 1214
        self.target_model = target_model
        # if target_model == 'hubert_base':
        #     hidden_dim = int(source_channel_size * hidden_size_factor)
        # elif target_model == 'mert_v0_public':
        #     hidden_dim = math.floor(int(source_channel_size * hidden_size_factor * 3 / 2))
        # elif target_model =='ast':
        #     hidden_dim = 1050
        # Define the convolutional adapter
        self.adapter = nn.Sequential(
            nn.LayerNorm(source_channel_size),  # Apply LayerNorm over the channel dimension
            Rearrange('b t h c -> b c h t'),  # Permute to [B, C, H, T]
            nn.Conv2d(source_channel_size, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),  # Conv maintains shape
            nn.ReLU(),
            Rearrange('b c h t -> b t h c'),
            nn.LayerNorm(hidden_dim),  # Apply LayerNorm on the hidden dimension
            Rearrange('b c h t -> b t h c'),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),  # Maintains shape
            nn.ReLU(),
            Rearrange('b c h t -> b t h c'),
            nn.LayerNorm(hidden_dim),  # Apply LayerNorm on the hidden dimension
            Rearrange('b c h t -> b t h c'),
            nn.Conv2d(hidden_dim, source_channel_size, kernel_size=(1, 3), padding=(0, 1)),  # Return to original size
            Rearrange('b c h t -> b t h c'),
        )
        
        self.length_adjuster = nn.Sequential(
            nn.LayerNorm(source_channel_size),  # Apply LayerNorm over the last dimension
            
            Rearrange('b c h t -> b t h c'),
            nn.Conv2d(source_channel_size, source_channel_size, kernel_size=(1, 3), padding=(0, 1)),  # Conv along the T dimension
            Rearrange('b c h t -> b t h c'),

            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.target_length, source_channel_size)),  # Resize to target length
        )
        
        # self.adapter.register_full_backward_hook(self.grad_divide_hook)
        # self.length_adjuster.register_full_backward_hook(self.grad_divide_hook)
        
    def grad_divide_hook(self, module, grad_input, grad_output):
        """Hook function to divide the gradient by a factor (e.g., 5)."""
        factor = 5  # The factor by which to divide the gradients
        # Make sure the number of grad_input elements matches exactly what is expected
        if len(grad_input) == 1:  # Typically for nn.Linear
            return (grad_input[0] / factor,)  # Return as a single-element tuple
        else:
            return tuple(grad / factor if grad is not None else None for grad in grad_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LightConvAdapterHead.

        Args:
            x (torch.Tensor): input tensor with shape [B, T, H, C].

        Returns:
            torch.Tensor: transformed tensor matching the input shape.
        """
        x = self.adapter(x)
        # if self.target_model == 'ast':
        #     x = self.length_adjuster(x)
        return x


class TransformerAdapterHead(nn.Module):
    """Transformer Adapter module.

    Transforms features from source size [B, T, H, C] to target size [B, T, H, C].
    Uses a Transformer to map temporal dependencies while preserving spatial features.
    """

    def __init__(
        self,
        source_size: Union[Tuple[int, ...], torch.Size],
        target_model: str,
        hidden_size_factor: Union[int, float] = 1.0,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
    ):
        """Initialization function for TransformerAdapterHead.

        Args:
            source_size (Union[Tuple[int, ...], torch.Size]): the size of the source feature.
            hidden_size_factor (Union[int, float]): the size of hidden dim of feature translator as a factor of input feature hidden dim.
            num_layers (int): number of transformer layers.
            num_heads (int): number of attention heads in the transformer.
            ff_dim (int): hidden size of the feed-forward network within the transformer.
        """
        super().__init__()
        self.hidden_size_factor = hidden_size_factor
        source_channel_size = source_size[-1]  # Feature dimension (C)
        hidden_dim = int(source_channel_size * hidden_size_factor)  # Adjust based on the channel dimension

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=source_channel_size,  # Feature dimension (C)
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(source_channel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for TransformerAdapterHead.

        Args:
            x (torch.Tensor): input tensor with shape [B, T, H, C].

        Returns:
            torch.Tensor: transformed tensor matching the input shape.
        """
        # Rearrange input for transformer: [B, T, H, C] -> [B, T, C]
        x = x.permute(0, 1, 3, 2).contiguous()

        # Apply the transformer encoder
        x = self.transformer_encoder(x)

        # Apply LayerNorm
        x = self.layer_norm(x)

        # Rearrange back to original shape: [B, T, C] -> [B, T, H, C]
        x = x.permute(0, 1, 2, 3)

        # Adjust length if necessary (e.g., for AST model)

        return x

