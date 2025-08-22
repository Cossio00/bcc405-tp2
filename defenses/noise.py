import torch

def add_noise(gradients, sigma=1e-3):
    return [g + sigma*torch.randn_like(g) for g in gradients]
