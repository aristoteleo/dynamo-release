from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from equiformer_pytorch import Equiformer


from typing import Optional, Mapping, Tuple, Union
from torch import nn, Tensor

from .abstract_vf import VelocityFieldBase

Number = Union[float, int, Tensor]


class EquiformerPointHead(nn.Module):
    """
    Equiformer-based point cloud processing head.

    Input:
      feats:  N x D_in or 1 x N x D_in  (continuous scalar features)
      coors:  N x 3   or 1 x N x 3      (spatial coordinates)
      mask:   N or 1 x N (bool, optional) (validity mask for points)

    Output:
      pred_scalar: N x D_out   (rotation/translation invariant, used as dz/dt)
      pred_vector: N x 3       (rotation equivariant, used as dp/dt)
    """
    def __init__(
        self,
        d_in: int,                 # Input scalar dimension
        c0: int = None,            # Backbone l=0 channel count, defaults to d_in
        c1: int = 64,              # Backbone l=1 channel count
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        d_out: Optional[int] = None,     # Output scalar dimension (defaults to d_in)
        reversible: bool = True,
        num_neighbors: int = 16,
        attend_self: bool = True,
        reduce_dim_out: bool = False,
        l2_dist_attention: bool = False,
    ):
        """
        Initialize EquiformerPointHead.

        Args:
            d_in: Input scalar feature dimension.
            c0: Number of l=0 (scalar) channels in backbone. Defaults to d_in if None.
            c1: Number of l=1 (vector) channels in backbone. Defaults to 64.
            depth: Number of Equiformer layers. Defaults to 4.
            heads: Number of attention heads. Defaults to 4.
            dim_head: Dimension per attention head. Defaults to 32.
            d_out: Output scalar dimension. Defaults to d_in if None.
            reversible: Whether to use reversible layers for memory efficiency. Defaults to True.
            num_neighbors: Number of nearest neighbors for local attention. Defaults to 16.
            attend_self: Whether to include self-attention. Defaults to True.
            reduce_dim_out: Whether to reduce output dimensions. Defaults to False.
            l2_dist_attention: Whether to use L2 distance-based attention. Defaults to False.
        """
        super().__init__()
        if c0 is None:
            c0 = d_in
        if d_out is None:
            d_out = d_in

        self.backbone = Equiformer(
            dim = (c0, c1),
            dim_head = (dim_head, dim_head),
            heads = (heads, heads),
            num_linear_attn_heads = 0,
            num_degrees = 2,
            depth = depth,
            attend_self = attend_self,
            reduce_dim_out = reduce_dim_out,
            l2_dist_attention = l2_dist_attention,
            reversible = reversible,
            num_neighbors = num_neighbors,
        )

        # Scalar head: l=0 -> d_out
        self.scalar_head = nn.Linear(c0, d_out)

        # Vector head: weighted sum of c1 vector channels at l=1 -> 1 vector channel (3D)
        self.vector_head = nn.Linear(c1, 1, bias=False)

    @staticmethod
    def _ensure_batched(x: torch.Tensor, last_dim: Optional[int] = None):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # -> 1 x N x D
        if last_dim is not None and x.size(-1) != last_dim:
            raise ValueError(f"Expected last dim {last_dim}, got {x.size(-1)}")
        return x

    def forward(
        self,
        feats: torch.Tensor,         # N x D_in or 1 x N x D_in
        coors: torch.Tensor,         # N x 3   or 1 x N x 3
        mask: Optional[torch.Tensor] = None  # N or 1 x N, bool
    ):
        """
        Forward pass through EquiformerPointHead.

        Args:
            feats: Input scalar features, shape (N, D_in) or (1, N, D_in).
            coors: Spatial coordinates, shape (N, 3) or (1, N, 3).
            mask: Optional validity mask for points, shape (N,) or (1, N), bool tensor.
                 If None, all points are considered valid.

        Returns:
            A tuple of (pred_scalar, pred_vector) where:
                - pred_scalar: Predicted scalar features, shape (N, d_out).
                - pred_vector: Predicted vector features, shape (N, 3).
        """
        device = feats.device
        feats = self._ensure_batched(feats)
        coors = self._ensure_batched(coors, 3)

        if mask is None:
            mask = torch.ones(feats.shape[:2], dtype=torch.bool, device=device)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # -> 1 x N
            mask = mask.to(device)

        out = self.backbone(feats, coors, mask)   # out.type0: 1 x N x c0; out.type1: 1 x N x c1 x 3

        pred_scalar = self.scalar_head(out.type0)  # 1 x N x d_out
        w = self.vector_head.weight.view(1, 1, -1, 1)  # 1 x 1 x c1 x 1
        pred_vector = (out.type1 * w).sum(dim=2)       # 1 x N x 3

        return pred_scalar.squeeze(0), pred_vector.squeeze(0)  # N x d_out, N x 3



class EFVelocity(VelocityFieldBase):
    """
    Equiformer-based velocity field for spatiotemporal dynamics.

    Input:
        z: Latent attributes, shape (N, z_dim)
        p: Spatial positions, shape (N, 3)
        t: Time scalar

    Output:
        dz_dt: Time derivative of latent attributes, shape (N, z_dim)
        dp_dt: Time derivative of spatial positions, shape (N, 3)
    """
    def __init__(
        self,
        z_dim: int,
        p_dim: int = 3,
        *,
        use_time: bool = True,
        predict_p: bool = True,
        strict_checks: bool = False,
        # EquiformerPointHead hyperparameters
        c0: int | None = None,
        c1: int = 64,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        reversible: bool = True,
        num_neighbors: int = 16,
        attend_self: bool = True,
        reduce_dim_out: bool = False,
        l2_dist_attention: bool = False,
    ):
        """
        Initialize EFVelocity.

        Args:
            z_dim: Dimension of latent attribute vector z.
            p_dim: Dimension of position vector p. Defaults to 3 for 3D space.
            use_time: Whether to concatenate time t as an additional scalar channel to z.
                     Defaults to True.
            predict_p: Whether to predict dp/dt (spatial velocity). If False, dp/dt is set to zero.
                      Defaults to True.
            strict_checks: If True, performs additional validation checks. Defaults to False.
            c0: Number of l=0 (scalar) channels in Equiformer backbone.
                If None, defaults to (z_dim + 1) if use_time else z_dim.
            c1: Number of l=1 (vector) channels in Equiformer backbone. Defaults to 64.
            depth: Number of Equiformer layers. Defaults to 4.
            heads: Number of attention heads. Defaults to 4.
            dim_head: Dimension per attention head. Defaults to 32.
            reversible: Whether to use reversible layers for memory efficiency. Defaults to True.
            num_neighbors: Number of nearest neighbors for local attention. Defaults to 16.
            attend_self: Whether to include self-attention. Defaults to True.
            reduce_dim_out: Whether to reduce output dimensions. Defaults to False.
            l2_dist_attention: Whether to use L2 distance-based attention. Defaults to False.
        """
        super().__init__(z_dim=z_dim, p_dim=p_dim, strict_checks=strict_checks)
        self.use_time = use_time
        self.predict_p = predict_p

        d_in = z_dim + (1 if use_time else 0)  # Concatenate z and t as additional scalar channel
        self.head = EquiformerPointHead(
            d_in = d_in,
            c0 = (c0 if c0 is not None else d_in),
            c1 = c1,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            d_out = z_dim,                 # Scalar head output dimension aligned with z_dim -> dz/dt
            reversible = reversible,
            num_neighbors = num_neighbors,
            attend_self = attend_self,
            reduce_dim_out = reduce_dim_out,
            l2_dist_attention = l2_dist_attention,
        )

    def _time_cat(self, z: Tensor, t: Number) -> Tensor:
        if not self.use_time:
            return z
        t_tensor = t if isinstance(t, Tensor) else torch.tensor(t, dtype=z.dtype, device=z.device)
        t_feat = t_tensor.reshape(1, 1).expand(z.size(0), 1)  # N x 1
        return torch.cat([z, t_feat], dim=-1)

    def _compute_velocity(
        self,
        z: Tensor,
        p: Tensor,
        t: Number,
        context: Optional[Mapping] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute velocity using Equiformer-based architecture.

        Args:
            z: Latent attribute tensor, shape (N, z_dim).
            p: Position tensor, shape (N, p_dim).
            t: Time scalar.
            context: Optional dictionary that may contain:
                    - 'mask': Validity mask for points, shape (N,) or (1, N), bool tensor.

        Returns:
            A tuple of (dz_dt, dp_dt) where:
                - dz_dt: Time derivative of latent attributes, shape (N, z_dim).
                - dp_dt: Time derivative of positions, shape (N, p_dim).
                        If predict_p is False, returns zeros.
        """
        feats = self._time_cat(z, t)
        mask = None
        if context is not None:
            mask = context.get("mask", None)  # Can pass N or 1xN bool mask

        dz_dt, dp_dt_pred = self.head(feats, p, mask)

        if not self.predict_p:
            dp_dt = torch.zeros_like(p)
        else:
            dp_dt = dp_dt_pred  # Directly interpret as dp/dt (equivariant vector field)

        return dz_dt, dp_dt


