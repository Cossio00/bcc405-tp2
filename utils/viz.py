import matplotlib.pyplot as plt
import torch

def plot_recon(x_true, x_recon, path):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x_true[0].squeeze(), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(x_recon[0].squeeze(), cmap='gray')
    axs[1].set_title('Reconstruída')
    plt.savefig(path)
    plt.close()