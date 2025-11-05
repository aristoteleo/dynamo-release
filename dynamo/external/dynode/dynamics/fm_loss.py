import torch

def sample_s(sampler="uniform", beta_alpha=0.5):
    """
    Sample interpolation parameter s from specified distribution.

    Args:
        sampler: Sampling method, either 'uniform' or 'beta'. Defaults to 'uniform'.
        beta_alpha: Alpha and beta parameters for Beta distribution (used when sampler='beta').
                   Defaults to 0.5.

    Returns:
        A scalar value in [0, 1] sampled from the specified distribution.

    Raises:
        ValueError: If sampler is not 'uniform' or 'beta'.
    """
    if sampler == "uniform":
        return torch.rand(1, 1).item()
    elif sampler == "beta":
        dist = torch.distributions.Beta(beta_alpha, beta_alpha)
        return dist.sample((1, 1)).item()
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

def flow_matching_loss(
    model,
    x_s, x_t,        # [B,D] or [B,N,D]
    t_s, t_t,        # [], single value tensors
    sampler="uniform",
    beta_alpha=0.5,
    lambda_spatial=10.0,  # Spatial loss weight
    lambda_energy=1e-4,
    eps_dt=1e-8,
    mute_spatial_velo = False,
    **model_kwargs
):
    """
    Flow matching loss with linear constant-velocity interpolation.

    Uses Uniform/Beta sampling with energy regularization and separate losses
    for gene expression and spatial components.

    Args:
        model: The velocity field model to evaluate. Should accept (x, t, **kwargs)
               and return predicted velocity.
        x_s: Source state tensor, shape [B, D] or [B, N, D].
        x_t: Target state tensor, shape [B, D] or [B, N, D].
        t_s: Source time, scalar tensor.
        t_t: Target time, scalar tensor.
        sampler: Interpolation sampling method, either 'uniform' or 'beta'.
                Defaults to 'uniform'.
        beta_alpha: Parameter for Beta distribution (when sampler='beta'). Defaults to 0.5.
        lambda_spatial: Weight for spatial velocity loss. Used to balance dimension imbalance.
                       Recommended value: z_dim / 3 (e.g., ~16.67 when z_dim=50).
                       Defaults to 10.0.
        lambda_energy: Weight for energy regularization term. Defaults to 1e-4.
        eps_dt: Minimum time difference to avoid division by zero. Defaults to 1e-8.
        mute_spatial_velo: If True, only compute gene expression loss and ignore spatial loss.
                          Defaults to False.
        **model_kwargs: Additional keyword arguments to pass to the model.

    Returns:
        A tuple of (loss, stats) where:
            - loss: Total scalar loss value.
            - stats: Dictionary containing detailed loss components:
                * 'loss_total': Total loss
                * 'loss_vel': Total velocity matching loss
                * 'loss_vel_gene': Gene expression velocity loss
                * 'loss_vel_spatial': Spatial velocity loss
                * 'loss_energy': Total energy regularization
                * 'loss_energy_gene': Gene expression energy
                * 'loss_energy_spatial': Spatial energy
                * 'lambda_spatial': The spatial weight used
    """
    device = x_s.device
    B = x_s.shape[0]

    dt = (t_t - t_s).clamp_min(eps_dt)
    dx = x_t - x_s

    # Sample interpolation parameter s
    s = sample_s(sampler, beta_alpha)

    # Linear path with constant velocity target
    tau    = t_s + s * dt
    x_mid  = x_s + s * dx
    v_star = dx / dt

    # Model prediction
    v_pred = model(x_mid, tau.item(), **model_kwargs)

    # Separate gene expression and spatial components
    z_dim = x_s.shape[-1] - 3  # Assume last 3 dimensions are spatial coordinates

    dz_pred = v_pred[..., :z_dim]   # Gene expression velocity
    dp_pred = v_pred[..., -3:]      # Spatial position velocity

    dz_star = v_star[..., :z_dim]
    dp_star = v_star[..., -3:]

    # Loss computation
    if mute_spatial_velo:
        # Only compute gene expression loss
        L_vel_z = ((dz_pred - dz_star) ** 2).mean()
        L_vel_p = torch.tensor(0.0, device=device)
        L_vel = L_vel_z

        L_energy_z = (dz_pred ** 2).mean()
        L_energy_p = torch.tensor(0.0, device=device)
        L_energy = L_energy_z
    else:
        # Compute gene and spatial losses separately
        L_vel_z = ((dz_pred - dz_star) ** 2).mean()
        L_vel_p = ((dp_pred - dp_star) ** 2).mean()
        L_vel = L_vel_z + lambda_spatial * L_vel_p

        L_energy_z = (dz_pred ** 2).mean()
        L_energy_p = (dp_pred ** 2).mean()
        L_energy = L_energy_z + lambda_spatial * L_energy_p

    loss = L_vel + lambda_energy * L_energy

    stats = {
        "loss_total": loss.detach(),
        "loss_vel": L_vel.detach(),
        "loss_vel_gene": L_vel_z.detach(),
        "loss_vel_spatial": L_vel_p.detach(),
        "loss_energy": L_energy.detach(),
        "loss_energy_gene": L_energy_z.detach(),
        "loss_energy_spatial": L_energy_p.detach(),
        "lambda_spatial": lambda_spatial,
    }
    return loss, stats

