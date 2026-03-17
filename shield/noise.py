"""Diffusion noise injection for contrastive decoding (VCD-compatible)."""

import torch


def add_diffusion_noise(image_tensor, noise_step):
    """Add diffusion-schedule noise to an image tensor."""
    num_steps = 1000

    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise

    noisy_image = image_tensor.clone()
    return q_x(noisy_image, noise_step)
