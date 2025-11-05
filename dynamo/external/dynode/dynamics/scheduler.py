"""
Dynamic weight scheduler for balancing spatial and gene expression losses.
"""
import torch


class SpatialWeightScheduler:
    """
    Scheduler for dynamically adjusting spatial loss weight.

    Starts with a smaller weight during early training (when gene expression is easier to learn),
    then gradually increases to the target weight to balance spatial and gene loss contributions.

    Args:
        z_dim: Dimension of gene expression features.
        p_dim: Dimension of spatial positions. Defaults to 3.
        warmup_epochs: Number of warmup epochs during which the weight gradually increases
                      from 1.0 to the target value. Defaults to 10.
        target_ratio: Target weight relative to the theoretical value. Defaults to 1.0.
                     Theoretical value = z_dim / p_dim
                     Can be adjusted to fine-tune the balance.

    Example:
        >>> scheduler = SpatialWeightScheduler(z_dim=50, warmup_epochs=10)
        >>> for epoch in range(100):
        ...     lambda_spatial = scheduler.step()
        ...     # use lambda_spatial in loss calculation
    """

    def __init__(
        self,
        z_dim: int,
        p_dim: int = 3,
        warmup_epochs: int = 10,
        target_ratio: float = 1.0,
    ):
        self.z_dim = z_dim
        self.p_dim = p_dim
        self.warmup_epochs = warmup_epochs
        self.target_ratio = target_ratio

        # Theoretical weight to make both loss components have similar magnitudes
        self.target_weight = (z_dim / p_dim) * target_ratio

        # Current epoch counter
        self.current_epoch = 0

        print(f"SpatialWeightScheduler initialized:")
        print(f"  z_dim={z_dim}, p_dim={p_dim}")
        print(f"  target_weight={self.target_weight:.2f}")
        print(f"  warmup_epochs={warmup_epochs}")

    def step(self) -> float:
        """
        Get the spatial weight for the current epoch and increment the epoch counter.

        Returns:
            lambda_spatial: Current spatial loss weight.
        """
        if self.current_epoch < self.warmup_epochs:
            # Linear increase from 1.0 to target_weight
            alpha = self.current_epoch / self.warmup_epochs
            weight = 1.0 + alpha * (self.target_weight - 1.0)
        else:
            # Maintain target weight after warmup
            weight = self.target_weight

        self.current_epoch += 1
        return weight

    def get_weight(self, epoch: int = None) -> float:
        """
        Get the weight for a specified epoch without changing the internal counter.

        Args:
            epoch: The specified epoch number. If None, uses the current epoch.

        Returns:
            lambda_spatial: Spatial loss weight for the specified epoch.
        """
        if epoch is None:
            epoch = self.current_epoch

        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            weight = 1.0 + alpha * (self.target_weight - 1.0)
        else:
            weight = self.target_weight

        return weight

    def reset(self):
        """Reset the epoch counter to zero."""
        self.current_epoch = 0


class AdaptiveSpatialWeightScheduler:
    """
    Adaptive spatial loss weight scheduler that adjusts dynamically based on actual loss ratios.

    Args:
        z_dim: Dimension of gene expression features.
        p_dim: Dimension of spatial positions. Defaults to 3.
        target_ratio: Target ratio of spatial_loss/gene_loss. Defaults to 1.0 for balance.
        adjustment_rate: Rate of adjustment. Defaults to 0.1.
        min_weight: Minimum weight value. Defaults to 1.0.
        max_weight: Maximum weight value. Defaults to 100.0.

    Example:
        >>> scheduler = AdaptiveSpatialWeightScheduler(z_dim=50)
        >>> for epoch in range(100):
        ...     loss, stats = flow_matching_loss(...)
        ...     lambda_spatial = scheduler.step(
        ...         gene_loss=stats['loss_vel_gene'],
        ...         spatial_loss=stats['loss_vel_spatial']
        ...     )
    """

    def __init__(
        self,
        z_dim: int,
        p_dim: int = 3,
        target_ratio: float = 1.0,
        adjustment_rate: float = 0.1,
        min_weight: float = 1.0,
        max_weight: float = 100.0,
    ):
        self.z_dim = z_dim
        self.p_dim = p_dim
        self.target_ratio = target_ratio
        self.adjustment_rate = adjustment_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initial weight
        self.current_weight = z_dim / p_dim

        print(f"AdaptiveSpatialWeightScheduler initialized:")
        print(f"  initial_weight={self.current_weight:.2f}")
        print(f"  target_ratio={target_ratio}")

    def step(self, gene_loss: float, spatial_loss: float) -> float:
        """
        Adjust the weight based on current loss ratios.

        Args:
            gene_loss: Gene expression loss value.
            spatial_loss: Spatial position loss value (unweighted).

        Returns:
            lambda_spatial: Updated spatial loss weight.
        """
        if spatial_loss < 1e-8:
            return self.current_weight

        # Current weighted ratio
        current_ratio = (spatial_loss * self.current_weight) / (gene_loss + 1e-8)

        # If spatial part is too small, increase weight; if too large, decrease weight
        if current_ratio < self.target_ratio:
            # Need to increase spatial weight
            adjustment = self.adjustment_rate * (self.target_ratio / (current_ratio + 1e-8) - 1.0)
            self.current_weight = self.current_weight * (1.0 + adjustment)
        else:
            # Need to decrease spatial weight
            adjustment = self.adjustment_rate * (1.0 - current_ratio / self.target_ratio)
            self.current_weight = self.current_weight * (1.0 + adjustment)

        # Constrain to reasonable range
        self.current_weight = max(self.min_weight, min(self.max_weight, self.current_weight))

        return self.current_weight

    def get_weight(self) -> float:
        """Get the current weight value."""
        return self.current_weight
