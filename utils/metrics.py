import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def mse(x, y):
    return F.mse_loss(x, y).item()

def psnr(x, y):
    mse_val = F.mse_loss(x, y).item()
    return 20 * np.log10(1.0 / np.sqrt(mse_val + 1e-8))

def compute_ssim(x, y):
    x_np = x.squeeze().detach().cpu().numpy()
    y_np = y.squeeze().detach().cpu().numpy()
    return ssim(x_np, y_np, data_range=1.0)
