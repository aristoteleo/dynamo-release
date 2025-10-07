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
    输入
      feats:  N x D_in or 1 x N x D_in  (连续标量特征)
      coors:  N x 3   or 1 x N x 3
      mask:   N or 1 x N (bool, 可选)
    输出
      pred_scalar: N x D_out   (旋转/平移不变, 用作 dz/dt)
      pred_vector: N x 3       (旋转等变,   用作 dp/dt)
    """
    def __init__(
        self,
        d_in: int,                 # 输入标量维度
        c0: int = None,            # 主干 l=0 通道数，默认 d_in
        c1: int = 64,              # 主干 l=1 通道数
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        d_out: Optional[int] = None,     # <--- 新增：标量输出维度（默认与 d_in 相同）
        reversible: bool = True,
        num_neighbors: int = 16,
        attend_self: bool = True,
        reduce_dim_out: bool = False,
        l2_dist_attention: bool = False,
    ):
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

        # 标量头：l=0 -> d_out
        self.scalar_head = nn.Linear(c0, d_out)

        # 向量头：对 l=1 的 c1 个向量通道做加权求和 -> 1 个向量通道 (3D)
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
        feats: torch.Tensor,         # N x D_in 或 1 x N x D_in
        coors: torch.Tensor,         # N x 3   或 1 x N x 3
        mask: Optional[torch.Tensor] = None  # N 或 1 x N，bool
    ):
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
    Equiformer-based velocity field
      输入:  z(N, z_dim), p(N, 3), t(scalar)
      输出:  dz_dt(N, z_dim), dp_dt(N, 3)
    """
    def __init__(
        self,
        z_dim: int,
        p_dim: int = 3,
        *,
        use_time: bool = True,
        predict_p: bool = True,
        strict_checks: bool = False,
        # EquiformerPointHead 超参数
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
        super().__init__(z_dim=z_dim, p_dim=p_dim, strict_checks=strict_checks)
        self.use_time = use_time
        self.predict_p = predict_p

        d_in = z_dim + (1 if use_time else 0)  # z 拼接 t 作为额外标量通道
        self.head = EquiformerPointHead(
            d_in = d_in,
            c0 = (c0 if c0 is not None else d_in),
            c1 = c1,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            d_out = z_dim,                 # 标量头输出维度对齐 z_dim -> dz/dt
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
        feats = self._time_cat(z, t)
        mask = None
        if context is not None:
            mask = context.get("mask", None)  # 可传 N 或 1xN 的 bool

        dz_dt, dp_dt_pred = self.head(feats, p, mask)

        if not self.predict_p:
            dp_dt = torch.zeros_like(p)
        else:
            dp_dt = dp_dt_pred  # 直接解释为 dp/dt（等变向量场）

        return dz_dt, dp_dt


