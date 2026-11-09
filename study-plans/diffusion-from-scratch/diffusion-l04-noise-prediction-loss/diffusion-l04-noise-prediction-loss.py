import torch

def noise_prediction_loss(x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss as a Python float under "loss" and the per-sample losses of shape (B,) under "per_sample_loss".
    """
    num_steps = len(alpha_bars)

    x0 = x0.flatten(start_dim=1)
    noise = noise.flatten(start_dim=1)
    alpha_bars = alpha_bars[timesteps,None]
    # forward process
    x_t = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * noise

    # predicting noise
    timesteps = timesteps[...,None].to(x0.dtype)
    noise_pred = x_t @ weight + timesteps / num_steps * time_weight[None,...] + bias[None,...]

    # calculate loss
    loss_per_sample = torch.mean((noise_pred - noise) ** 2, dim=1)
    loss = torch.mean(loss_per_sample)
    
    return {
        "loss": float(loss.item()),
        "per_sample_loss": loss_per_sample
    }
    
    
