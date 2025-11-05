"""
Simplified abstract base classes for embedding models.
Provides a clean contract for AE, VAE, CVAE and other embedding strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import torch
import torch.nn as nn


class BaseEmbeddingModel(nn.Module, ABC):
    """
    Abstract base class for all embedding models.
    
    Supports:
    - Standard Autoencoders (AE)
    - Variational Autoencoders (VAE)  
    - Conditional models (CVAE, Conditional AE)
    
    The condition tensor enables flexible conditioning:
    - Cell type labels (one-hot)
    - Continuous covariates
    - Multi-modal conditions
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,
        condition_dim: Optional[int] = None,
    ):
        """
        Initialize base embedding model.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            output_dim: Output dimension (defaults to input_dim)
            condition_dim: Conditioning vector dimension (for conditional models)
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim or input_dim
        self.condition_dim = condition_dim
    
    @abstractmethod
    def encode(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            condition: Optional conditioning tensor of shape (batch_size, condition_dim)
        
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        pass
    
    @abstractmethod
    def decode(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            condition: Optional conditioning tensor of shape (batch_size, condition_dim)
        
        Returns:
            Reconstructed data of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            condition: Optional conditioning tensor
        
        Returns:
            Dictionary containing at minimum:
                - 'reconstruction': Reconstructed output
                - 'latent': Latent representation
            May also include:
                - 'mu', 'log_var': For VAE models
        """
        pass
    
    @abstractmethod
    def loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the model.
        
        Args:
            x: Original input tensor
            outputs: Output dictionary from forward pass
            **kwargs: Additional loss arguments (e.g., beta for β-VAE)
        
        Returns:
            Dictionary containing:
                - 'total': Total loss (for backprop)
                - 'reconstruction': Reconstruction loss
                - Other components (e.g., 'kl' for VAE)
        """
        pass


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoder networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        condition_dim: Optional[int] = None,
    ):
        """
        Initialize encoder.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension  
            condition_dim: Optional conditioning dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            condition: Optional conditioning tensor
        
        Returns:
            For deterministic: latent tensor
            For probabilistic: dict with 'mu' and 'log_var'
        """
        pass


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for decoder networks.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        condition_dim: Optional[int] = None,
    ):
        """
        Initialize decoder.
        
        Args:
            latent_dim: Latent dimension
            output_dim: Output dimension
            condition_dim: Optional conditioning dimension
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
    
    @abstractmethod
    def forward(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor
            condition: Optional conditioning tensor
        
        Returns:
            Reconstructed output
        """
        pass