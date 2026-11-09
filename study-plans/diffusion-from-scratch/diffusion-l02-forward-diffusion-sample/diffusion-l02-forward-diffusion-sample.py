import torch

def forward_diffusion_sample(x0: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Returns the noisy samples x_t with the same shape and dtype as x0.
    """
    alpha_bars = alpha_bars[timesteps]
    alpha_bars = alpha_bars.reshape((len(x0),) + (1,) * (x0.ndim - 1)) 
    x_t = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * noise
    return x_t
