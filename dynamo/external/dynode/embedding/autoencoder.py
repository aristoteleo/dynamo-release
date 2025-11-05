"""
Autoencoder implementation based on abstract base classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from ..utils.ffn import FFN
from .abstract_ae import BaseEmbeddingModel, BaseEncoder, BaseDecoder


class FFNEncoder(BaseEncoder):
    """Encoder using FFN modules."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
        spectral_norm: bool = False,
        condition_dim: Optional[int] = None,
    ):
        super().__init__(input_dim, latent_dim, condition_dim)
        
        # Adjust input dimension if conditioning is used
        actual_input_dim = input_dim
        if condition_dim is not None:
            actual_input_dim += condition_dim
        
        if hidden_dims and len(hidden_dims) > 0:
            # Multi-layer encoder
            self.layers = nn.ModuleList()
            prev_dim = actual_input_dim
            
            for hidden_dim in hidden_dims:
                self.layers.append(FFN(
                    d_model=prev_dim,
                    hidden_dim=hidden_dim,
                    out_dim=hidden_dim,
                    kind=activation,
                    dropout=dropout,
                    bias=bias,
                    spectral_norm=spectral_norm,
                ))
                prev_dim = hidden_dim
            
            # Final layer to latent
            self.fc_latent = nn.Linear(prev_dim, latent_dim, bias=bias)
        else:
            # Single FFN encoder
            self.encoder_ffn = FFN(
                d_model=actual_input_dim,
                out_dim=latent_dim,
                kind=activation,
                dropout=dropout,
                bias=bias,
                spectral_norm=spectral_norm,
            )
            self.layers = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Concatenate conditioning if provided
        if condition is not None and self.condition_dim is not None:
            x = torch.cat([x, condition], dim=-1)
        
        if self.layers is not None:
            # Multi-layer encoder
            h = x
            for layer in self.layers:
                h = layer(h)
            return self.fc_latent(h)
        else:
            # Single FFN encoder
            return self.encoder_ffn(x)


class FFNDecoder(BaseDecoder):
    """Decoder using FFN modules."""
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
        spectral_norm: bool = False,
        condition_dim: Optional[int] = None,
    ):
        super().__init__(latent_dim, output_dim, condition_dim)
        
        # Adjust input dimension if conditioning is used
        actual_input_dim = latent_dim
        if condition_dim is not None:
            actual_input_dim += condition_dim
        
        if hidden_dims and len(hidden_dims) > 0:
            # Multi-layer decoder
            self.layers = nn.ModuleList()
            prev_dim = actual_input_dim
            
            for hidden_dim in hidden_dims:
                self.layers.append(FFN(
                    d_model=prev_dim,
                    hidden_dim=hidden_dim,
                    out_dim=hidden_dim,
                    kind=activation,
                    dropout=dropout,
                    bias=bias,
                    spectral_norm=spectral_norm,
                ))
                prev_dim = hidden_dim
            
            # Final layer to output
            self.fc_output = nn.Linear(prev_dim, output_dim, bias=bias)
        else:
            # Single FFN decoder
            self.decoder_ffn = FFN(
                d_model=actual_input_dim,
                out_dim=output_dim,
                kind=activation,
                dropout=dropout,
                bias=bias,
                spectral_norm=spectral_norm,
            )
            self.layers = None
    
    def forward(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Concatenate conditioning if provided
        if condition is not None and self.condition_dim is not None:
            z = torch.cat([z, condition], dim=-1)
        
        if self.layers is not None:
            # Multi-layer decoder
            h = z
            for layer in self.layers:
                h = layer(h)
            return self.fc_output(h)
        else:
            # Single FFN decoder
            return self.decoder_ffn(z)


class Autoencoder(BaseEmbeddingModel):
    """
    Standard Autoencoder implementation.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        output_dim: Output dimension (defaults to input_dim)
        encoder_hidden_dims: List of hidden dimensions for encoder
        decoder_hidden_dims: List of hidden dimensions for decoder
        activation: Activation type {"swiglu", "geglu", "gelu", "silu"}
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        spectral_norm: Whether to apply spectral normalization
        condition_dim: Conditioning vector dimension (for conditional AE)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        activation: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
        spectral_norm: bool = False,
        condition_dim: Optional[int] = None,
    ):
        super().__init__(input_dim, latent_dim, output_dim, condition_dim)
        
        self.encoder_module = FFNEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            condition_dim=condition_dim,
        )
        
        self.decoder_module = FFNDecoder(
            latent_dim=latent_dim,
            output_dim=self.output_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            condition_dim=condition_dim,
        )
    
    def encode(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder_module(x, condition)
    
    def decode(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder_module(z, condition)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        latent = self.encode(x, condition)
        reconstruction = self.decode(latent, condition)
        
        return {
            'reconstruction': reconstruction,
            'latent': latent,
        }
    
    def loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        reduction: str = 'mean',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        reconstruction = outputs['reconstruction']
        
        # Reconstruction loss (MSE)
        if reduction == 'mean':
            recon_loss = F.mse_loss(reconstruction, x)
        else:
            recon_loss = F.mse_loss(reconstruction, x, reduction='none')
            recon_loss = recon_loss.sum(dim=-1).mean()
        
        return {
            'total': recon_loss,
            'reconstruction': recon_loss,
        }