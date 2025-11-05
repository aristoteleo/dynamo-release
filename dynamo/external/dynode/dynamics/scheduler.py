"""
Dynamic weight scheduler for balancing spatial and gene expression losses
"""
import torch


class SpatialWeightScheduler:
    """
    动态调整空间损失权重的调度器

    在训练初期使用较小的权重（基因表达更容易学习），
    然后逐渐增加到目标权重，使空间和基因损失贡献平衡。

    Args:
        z_dim: 基因表达的维度
        p_dim: 空间位置的维度（默认3）
        warmup_epochs: 预热的epoch数，在这期间权重从1逐渐增加到目标值
        target_ratio: 目标权重相对于理论值的比例（默认1.0）
                     理论值 = z_dim / p_dim
                     可以调整这个比例来微调平衡

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

        # 理论上应该用的权重（使得两部分loss在同一量级）
        self.target_weight = (z_dim / p_dim) * target_ratio

        # 当前epoch计数
        self.current_epoch = 0

        print(f"SpatialWeightScheduler initialized:")
        print(f"  z_dim={z_dim}, p_dim={p_dim}")
        print(f"  target_weight={self.target_weight:.2f}")
        print(f"  warmup_epochs={warmup_epochs}")

    def step(self) -> float:
        """
        获取当前epoch的空间权重，并将epoch计数+1

        Returns:
            lambda_spatial: 当前的空间损失权重
        """
        if self.current_epoch < self.warmup_epochs:
            # 线性增长：从1.0到target_weight
            alpha = self.current_epoch / self.warmup_epochs
            weight = 1.0 + alpha * (self.target_weight - 1.0)
        else:
            # 预热结束后保持目标权重
            weight = self.target_weight

        self.current_epoch += 1
        return weight

    def get_weight(self, epoch: int = None) -> float:
        """
        获取指定epoch的权重（不改变内部计数）

        Args:
            epoch: 指定的epoch数，如果为None则使用当前epoch

        Returns:
            lambda_spatial: 空间损失权重
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
        """重置epoch计数"""
        self.current_epoch = 0


class AdaptiveSpatialWeightScheduler:
    """
    自适应调整空间损失权重，根据实际的loss比例动态调整

    Args:
        z_dim: 基因表达的维度
        p_dim: 空间位置的维度（默认3）
        target_ratio: 目标的spatial_loss/gene_loss比例（默认1.0表示平衡）
        adjustment_rate: 调整速率（默认0.1）
        min_weight: 最小权重（默认1.0）
        max_weight: 最大权重（默认100.0）

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

        # 初始权重
        self.current_weight = z_dim / p_dim

        print(f"AdaptiveSpatialWeightScheduler initialized:")
        print(f"  initial_weight={self.current_weight:.2f}")
        print(f"  target_ratio={target_ratio}")

    def step(self, gene_loss: float, spatial_loss: float) -> float:
        """
        根据当前的loss比例调整权重

        Args:
            gene_loss: 基因表达的loss值
            spatial_loss: 空间位置的loss值（未加权）

        Returns:
            lambda_spatial: 更新后的空间损失权重
        """
        if spatial_loss < 1e-8:
            return self.current_weight

        # 当前的加权比例
        current_ratio = (spatial_loss * self.current_weight) / (gene_loss + 1e-8)

        # 如果spatial部分太小，增加权重；如果太大，减小权重
        if current_ratio < self.target_ratio:
            # 需要增加spatial的权重
            adjustment = self.adjustment_rate * (self.target_ratio / (current_ratio + 1e-8) - 1.0)
            self.current_weight = self.current_weight * (1.0 + adjustment)
        else:
            # 需要减小spatial的权重
            adjustment = self.adjustment_rate * (1.0 - current_ratio / self.target_ratio)
            self.current_weight = self.current_weight * (1.0 + adjustment)

        # 限制在合理范围内
        self.current_weight = max(self.min_weight, min(self.max_weight, self.current_weight))

        return self.current_weight

    def get_weight(self) -> float:
        """获取当前权重"""
        return self.current_weight
