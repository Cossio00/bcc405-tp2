import torch

def clip_gradients(gradients, clip_value=1.0):
    return [torch.clamp(g, -clip_value, clip_value) for g in gradients]
