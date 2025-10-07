# velocity_field_base.py (default strict_checks=False)
from __future__ import annotations
import abc
from typing import Optional, Mapping, Tuple, Union
import torch
from torch import Tensor, nn

Number = Union[float, int, Tensor]

class VelocityFieldBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for velocity fields on point clouds with latent attributes.

    Conventions:
      - Concatenated input state: (N, z_dim + p_dim), with the **last p_dim dims = positions p**.
      - Split: state = [z, p], where z: (N, z_dim), p: (N, p_dim).
      - Output: concatenated [dz/dt, dp/dt] with same shape as state.
    """

    def __init__(self, z_dim: int, p_dim: int = 3, *, strict_checks: bool = False):
        super().__init__()
        self.z_dim = int(z_dim)
        self.p_dim = int(p_dim)
        self.strict_checks = bool(strict_checks)

    # -------- public API --------
    def forward(
        self,
        state: Tensor,        # (N, z_dim + p_dim); last p_dim are positions
        t: Number,            # scalar float or 0-d tensor
        context: Optional[Mapping] = None,
    ) -> Tensor:
        """
        Args:
          state: concatenated [z, p], last p_dim dims are p.
          t: time scalar in [0,1] (float or 0-d tensor)
          context: optional dict

        Returns:
          dstate_dt: concatenated [dz/dt, dp/dt], same shape as state
        """
        z, p = self._split_state(state)
        dz_dt, dp_dt = self._compute_velocity(z, p, t, context)
        dstate_dt = self._concat(dz_dt, dp_dt)
        if self.strict_checks:
            self._check_concat_shapes(state, dstate_dt)
        return dstate_dt

    def call_zp(
        self,
        z: Tensor,            # (N, z_dim)
        p: Tensor,            # (N, p_dim)
        t: Number,
        context: Optional[Mapping] = None,
    ) -> Tensor:
        """
        Convenience: accept (z, p) separately and return concatenated [dz/dt, dp/dt].
        """
        dz_dt, dp_dt = self._compute_velocity(z, p, t, context)
        return self._concat(dz_dt, dp_dt)

    # -------- subclass contract --------
    @abc.abstractmethod
    def _compute_velocity(
        self,
        z: Tensor,            # (N, z_dim)
        p: Tensor,            # (N, p_dim)
        t: Number,            # scalar
        context: Optional[Mapping] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Must return:
          dz_dt: (N, z_dim)
          dp_dt: (N, p_dim)
        """
        raise NotImplementedError

    # -------- utils --------
    def _split_state(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        z = state[..., : self.z_dim]
        p = state[..., -self.p_dim :]
        return z, p

    def _concat(self, dz_dt: Tensor, dp_dt: Tensor) -> Tensor:
        return torch.cat([dz_dt, dp_dt], dim=-1)

    # -------- checks (only if strict_checks=True) --------
    @torch.no_grad()
    def _check_concat_shapes(self, state: Tensor, dstate_dt: Tensor) -> None:
        if dstate_dt.shape != state.shape:
            raise ValueError(f"dstate/dt shape {tuple(dstate_dt.shape)} "
                             f"must match state shape {tuple(state.shape)}")
        if not torch.isfinite(dstate_dt).all():
            raise ValueError("dstate/dt contains NaN/Inf")
