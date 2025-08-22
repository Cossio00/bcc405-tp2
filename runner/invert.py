import os
import csv
import argparse
from torchvision.utils import save_image
import torch  # Assuming torch is used for x_true, x_recon, etc.

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run inversion experiment")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., mlp)")
parser.add_argument("--attack", type=str, required=True, help="Attack type (e.g., dlg)")
parser.add_argument("--defense", type=str, required=True, help="Defense type (e.g., none)")
parser.add_argument("--opt", type=str, required=True, help="Optimizer (e.g., L-BFGS)")
parser.add_argument("--scenario", type=str, required=True, help="Scenario (e.g., single)")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
parser.add_argument("--restarts", type=int, default=1, help="Number of restarts")
parser.add_argument("--clip", type=float, default=1.0, help="Clipping value")
parser.add_argument("--sigma", type=float, default=0.0, help="Sigma value")
args = parser.parse_args()

# -----------------------------
# Create directories
# -----------------------------
os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/figs", exist_ok=True)

# Output file names
exp_name = f"{args.model}_{args.attack}_{args.defense}_{args.opt}_{args.scenario}"
csv_path = f"results/csv/{exp_name}.csv"
fig_path = f"results/figs/{exp_name}.png"

# Save CSV (append if exists)
header = ["model", "attack", "defense", "opt", "scenario",
          "lr", "iters", "restarts", "clip", "sigma",
          "grad_loss", "mse", "psnr", "ssim", "time", "label_true", "label_recon"]

# Placeholder values (replace these with actual computations)
best_loss = 0.0  # Example placeholder
recon_mse = 0.0  # Example placeholder
recon_psnr = 0.0  # Example placeholder
recon_ssim = 0.0  # Example placeholder
elapsed = 0.0  # Example placeholder
y_true = torch.tensor([0, 1])  # Example placeholder
y_recon = torch.tensor([0, 1])  # Example placeholder
x_true = torch.randn(1, 3, 64, 64)  # Example placeholder
x_recon = torch.randn(1, 3, 64, 64)  # Example placeholder

row = [args.model, args.attack, args.defense, args.opt, args.scenario,
       args.lr, args.iters, args.restarts, args.clip, args.sigma,
       best_loss, recon_mse, recon_psnr, recon_ssim, elapsed,
       y_true.tolist(), y_recon.tolist()]

write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(row)

# Save reconstructed vs original image
save_image(torch.cat([x_true, x_recon], dim=0), fig_path, nrow=2, normalize=True)

print(f"\nResultados salvos em: {csv_path}")
print(f"Figura salva em: {fig_path}")