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
    lambda_energy=1e-4,
    eps_dt=1e-8,
    mute_spatial_velo = False,
    **model_kwargs
):
    """
    线性恒速插值 + Uniform/Beta 采样 + energy 正则
    返回: loss(标量), stats(dict)
    """
    device = x_s.device
    B = x_s.shape[0]

    #t_s = t_s.view(B, 1)
    #t_t = t_t.view(B, 1)
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

    # 损失
    if mute_spatial_velo:
        v_pred =v_pred[..., :-3]
        v_star =v_star[..., :-3]
        
    L_vel = ((v_pred - v_star) ** 2).mean()
    L_energy = (v_pred ** 2).mean()
    loss = L_vel + lambda_energy * L_energy

    stats = {
        "loss_total": loss.detach(),
        "loss_vel": L_vel.detach(),
        "loss_energy": L_energy.detach(),
    }
    return loss, stats

