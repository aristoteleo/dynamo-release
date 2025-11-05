# mlp_velocity.py
from __future__ import annotations
from typing import Optional, Mapping, Tuple, Union, Sequence
import torch
from torch import nn, Tensor

from .abstract_vf import VelocityFieldBase  

Number = Union[float, int, Tensor]

class StaticMLPVelocity(VelocityFieldBase):
    """
    Simplest velocity field using MLP.

    Predicts gene expression velocity while keeping spatial positions static:
      - dz/dt = MLP([z, t?]) with post-normalization
      - dp/dt = 0
    """

    def __init__(
        self,
        z_dim: int,
        p_dim: int = 3,
        hidden: Sequence[int] = (128, 128),
        use_time: bool = True,
        strict_checks: bool = False,
    ):
        """
        Initialize StaticMLPVelocity.

        Args:
            z_dim: Dimension of latent attribute vector z.
            p_dim: Dimension of position vector p. Defaults to 3 for 3D space.
            hidden: Sequence of hidden layer dimensions. Defaults to (128, 128).
            use_time: Whether to concatenate time t as an input feature. Defaults to True.
            strict_checks: If True, performs additional validation checks. Defaults to False.
        """
        super().__init__(z_dim=z_dim, p_dim=p_dim, strict_checks=strict_checks)
        self.use_time = use_time
        in_dim = z_dim + (1 if use_time else 0)

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.SiLU(),
                nn.LayerNorm(h)   # Post-normalization
            ]
            prev = h
        # Final output layer without normalization
        layers.append(nn.Linear(prev, z_dim))

        self.net = nn.Sequential(*layers)

    def _compute_velocity(
        self,
        z: Tensor,           # (N, z_dim)
        p: Tensor,           # (N, p_dim)
        t: Number,           # scalar
        context: Optional[Mapping] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute velocity using MLP. Spatial positions remain static.

        Args:
            z: Latent attribute tensor, shape (N, z_dim).
            p: Position tensor, shape (N, p_dim).
            t: Time scalar.
            context: Optional dictionary (unused in this implementation).

        Returns:
            A tuple of (dz_dt, dp_dt) where:
                - dz_dt: Time derivative of latent attributes, shape (N, z_dim).
                - dp_dt: Time derivative of positions, always zeros with shape (N, p_dim).
        """
        if self.use_time:
            t_tensor = t if isinstance(t, Tensor) else torch.tensor(
                t, dtype=z.dtype, device=z.device
            )
            t_feat = t_tensor.reshape(1, 1).expand(z.shape[0], 1)
            inp = torch.cat([z, t_feat], dim=-1)
        else:
            inp = z

        dz_dt = self.net(inp)               # (N, z_dim)
        dp_dt = torch.zeros_like(p)         # (N, p_dim) all zeros
        return dz_dt, dp_dt
