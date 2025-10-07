# mlp_velocity.py
from __future__ import annotations
from typing import Optional, Mapping, Tuple, Union, Sequence
import torch
from torch import nn, Tensor

from .abstract_vf import VelocityFieldBase  

Number = Union[float, int, Tensor]

class StaticMLPVelocity(VelocityFieldBase):
    """
    Simplest vector field:
      - dz/dt = MLP([z, t?]) with post-norm
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
        super().__init__(z_dim=z_dim, p_dim=p_dim, strict_checks=strict_checks)
        self.use_time = use_time
        in_dim = z_dim + (1 if use_time else 0)

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.SiLU(),
                nn.LayerNorm(h)   # <- PostNorm
            ]
            prev = h
        # 最后一层输出，不做归一化
        layers.append(nn.Linear(prev, z_dim))

        self.net = nn.Sequential(*layers)

    def _compute_velocity(
        self,
        z: Tensor,           # (N, z_dim)
        p: Tensor,           # (N, p_dim)
        t: Number,           # scalar
        context: Optional[Mapping] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.use_time:
            t_tensor = t if isinstance(t, Tensor) else torch.tensor(
                t, dtype=z.dtype, device=z.device
            )
            t_feat = t_tensor.reshape(1, 1).expand(z.shape[0], 1)
            inp = torch.cat([z, t_feat], dim=-1)
        else:
            inp = z

        dz_dt = self.net(inp)               # (N, z_dim)
        dp_dt = torch.zeros_like(p)         # (N, p_dim) 全零
        return dz_dt, dp_dt
