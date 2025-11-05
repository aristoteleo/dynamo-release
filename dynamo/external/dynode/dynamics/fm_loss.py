import torch

def sample_s(sampler="uniform", beta_alpha=0.5):
    if sampler == "uniform":
        return torch.rand(1, 1).item()
    elif sampler == "beta":
        dist = torch.distributions.Beta(beta_alpha, beta_alpha)
        return dist.sample((1, 1)).item()
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

def flow_matching_loss(
    model,
    x_s, x_t,        # [B,D] 或 [B,N,D]
    t_s, t_t,        # [], single value tensors
    sampler="uniform",
    beta_alpha=0.5,
    lambda_spatial=10.0,  # 新增：空间损失权重
    lambda_energy=1e-4,
    eps_dt=1e-8,
    mute_spatial_velo = False,
    **model_kwargs
):
    """
    线性恒速插值 + Uniform/Beta 采样 + energy 正则 + 分离损失
    返回: loss(标量), stats(dict)

    Args:
        lambda_spatial: 空间速度损失的权重，用于平衡维度不平衡问题
                       推荐值: z_dim / 3 (例如 z_dim=50 时约为 16.67)
    """
    device = x_s.device
    B = x_s.shape[0]

    dt = (t_t - t_s).clamp_min(eps_dt)
    dx = x_t - x_s

    # 采样 s 和 tau
    s = sample_s(sampler, beta_alpha)  # [B,1]

    # 线性路径与恒速目标
    tau    = t_s + s * dt
    x_mid  = x_s + s * dx
    v_star = dx / dt

    # 模型预测
    v_pred = model(x_mid, tau.item(), **model_kwargs)

    # 分离基因表达和空间部分
    z_dim = x_s.shape[-1] - 3  # 假设最后3维是空间坐标

    dz_pred = v_pred[..., :z_dim]   # 基因表达速度
    dp_pred = v_pred[..., -3:]      # 空间位置速度

    dz_star = v_star[..., :z_dim]
    dp_star = v_star[..., -3:]

    # 损失计算
    if mute_spatial_velo:
        # 只计算基因表达损失
        L_vel_z = ((dz_pred - dz_star) ** 2).mean()
        L_vel_p = torch.tensor(0.0, device=device)
        L_vel = L_vel_z

        L_energy_z = (dz_pred ** 2).mean()
        L_energy_p = torch.tensor(0.0, device=device)
        L_energy = L_energy_z
    else:
        # 分别计算基因和空间损失
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

