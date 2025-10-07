import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as _spectral_norm


def _round_to_multiple(x: int, base: int, mode: str = "nearest") -> int:
    """
    Round integer x to a multiple of `base`.

    Args:
        x: Value to round.
        base: The multiple to round to (e.g., 128, 256). If <=0, returns x.
        mode: One of {"nearest", "up", "down"}.

    Returns:
        int: Rounded value.
    """
    if base <= 0:
        return x
    if mode == "up":
        return int(math.ceil(x / base) * base)
    if mode == "down":
        return int(math.floor(x / base) * base)
    # nearest
    down = int(math.floor(x / base) * base)
    up = int(math.ceil(x / base) * base)
    return up if (x - down) > (up - x) else down


class FFN(nn.Module):
    """
    Flexible feed-forward network for Transformer blocks (no norm/residual inside).

    Design:
        - GLU family ("swiglu"/"geglu"): one Linear produces 2 * hidden_dim, split into
          value and gate, then element-wise multiply, followed by the output projection.
        - Plain two-layer FFN ("gelu"/"silu"): Linear -> activation -> Linear.
        - No normalization, residual, or DropPath in this module; those belong outside.

    Args:
        d_model (int):
            Input (and default output) channel size, typically the Transformer model width.
        hidden_dim (int | None):
            Intermediate width. If None, it is inferred based on `kind` and then aligned:
              * kind in {"gelu", "silu"}   -> target = 4 * d_model
              * kind in {"swiglu", "geglu"} -> target = (8/3) * d_model (≈ 2.67×)
            The target is rounded to the nearest multiple of `align_to`
            (default: 256 if d_model >= 1024 else 128).
        out_dim (int | None):
            Output channel size. Defaults to `d_model` (convenient for residual add).
        kind (str):
            Activation/topology: one of {"swiglu", "geglu", "gelu", "silu"}.
            - "swiglu": gate uses SiLU
            - "geglu" : gate uses GELU
            - "gelu"  : plain two-layer FFN with GELU
            - "silu"  : plain two-layer FFN with SiLU
        dropout (float):
            Dropout probability applied once on the FFN output. Default 0.0.
        bias (bool):
            Whether Linear layers use bias. Defaults to False (common in modern LLMs).
        spectral_norm (bool):
            If True, wrap Linear layers with spectral normalization. Default False.
            (Not typically needed; reserved for special stability constraints.)
        align_to (int | None):
            Multiple to align `hidden_dim` to when auto-inferred. If None, uses
            256 when d_model >= 1024, else 128.

    Attributes:
        d_model (int): Saved model width.
        hidden_dim (int): Final intermediate width after inference/alignment.
        out_dim (int): Final output width.
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        kind: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
        spectral_norm: bool = False,
        align_to: int | None = None,
    ):
        super().__init__()
        assert kind in {"swiglu", "geglu", "gelu", "silu"}

        self.d_model = d_model
        self.kind = kind
        self.out_dim = out_dim if out_dim is not None else d_model

        # --- Auto-infer hidden_dim when not provided ---
        if hidden_dim is None:
            base = align_to if align_to is not None else (256 if d_model >= 1024 else 128)
            if kind in {"swiglu", "geglu"}:
                target = int(round((8.0 / 3.0) * d_model))  # ≈2.67× d_model
            else:  # 'gelu' or 'silu'
                target = 4 * d_model
            hidden_dim = _round_to_multiple(target, base, mode="nearest")
        self.hidden_dim = hidden_dim

        # --- Layers ---
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity()

        if kind in {"swiglu", "geglu"}:
            # Produce 2*hidden_dim in one matmul; then split into (value, gate).
            lin_in = nn.Linear(d_model, 2 * hidden_dim, bias=bias)
            lin_out = nn.Linear(hidden_dim, self.out_dim, bias=bias)
        else:  # 'gelu' or 'silu'
            lin_in = nn.Linear(d_model, hidden_dim, bias=bias)
            lin_out = nn.Linear(hidden_dim, self.out_dim, bias=bias)

        if spectral_norm:
            lin_in = _spectral_norm(lin_in)
            lin_out = _spectral_norm(lin_out)

        self.proj_in = lin_in
        self.proj_out = lin_out

        # Activation: used as gate (GLU) or mid-activation (two-layer FFN).
        self.act = {
            "swiglu": F.silu,   # gate activation
            "geglu":  F.gelu,   # gate activation
            "gelu":   F.gelu,   # mid activation
            "silu":   F.silu,   # mid activation
        }[kind]

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier-uniform for Linear weights; zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Tensor of shape (..., out_dim). If out_dim == d_model, it fits residual add.
        """
        if self.kind in {"swiglu", "geglu"}:
            value, gate = self.proj_in(x).chunk(2, dim=-1)
            z = value * self.act(gate)      # GLU-style gating
        else:
            z = self.act(self.proj_in(x))   # plain two-layer FFN mid-activation
        out = self.proj_out(z)
        return self.dropout(out)