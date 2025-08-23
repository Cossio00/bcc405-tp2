import os
import csv
import argparse
import time
import torch
from torchvision.utils import save_image
from models.mlp import MLP
from models.cnn_small import SmallCNN
from attacks.dlg import DLGAttack
from attacks.idlg import IDLGAttack
from defenses.clipping import clip_gradients
from defenses.noise import add_noise

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run inversion experiment")
parser.add_argument("--model", type=str, required=True, choices=['mlp', 'cnn'])
parser.add_argument("--attack", type=str, required=True, choices=['dlg', 'idlg'])
parser.add_argument("--defense", type=str, default='none', choices=['none', 'clip', 'noise'])
parser.add_argument("--opt", type=str, required=True, choices=['adam', 'lbfgs'])
parser.add_argument("--scenario", type=str, required=True, choices=['single', 'batch'])
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--iters", type=int, default=10000, help="Number of iterations")  # Aumentado para 5000
parser.add_argument("--restarts", type=int, default=5, help="Number of restarts")
parser.add_argument("--clip", type=float, default=1.0, help="Clipping value")
parser.add_argument("--sigma", type=float, default=0.0, help="Sigma value") 
args = parser.parse_args()

# Create directories
os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/figs", exist_ok=True)

# Output file names
exp_name = f"{args.model}_{args.attack}_{args.defense}_{args.opt}_{args.scenario}"
csv_path = f"results/csv/{exp_name}.csv"
fig_path = f"results/figs/{exp_name}.png"

# Load model and data
device = torch.device("cpu")  # Adjust to 'cuda' if available
if args.model == 'mlp':
    model = MLP()
    input_shape = (1, 1, 28, 28) if args.scenario == 'single' else (4, 1, 28, 28)
elif args.model == 'cnn':
    model = SmallCNN()
    input_shape = (1, 1, 28, 28) if args.scenario == 'single' else (4, 1, 28, 28)
checkpoint = torch.load(f'results/grads/{args.model}_grads_{args.scenario}.pt')
model.load_state_dict(checkpoint['model_state'])
model.train()  # Habilitar gradientes
for param in model.parameters():
    param.requires_grad_(True)  # Garantir que todos os parâmetros aceitem gradientes
model.to(device)
grad_true = checkpoint['grads']
x_true, y_true = checkpoint['data']
if args.scenario == 'single':
    x_true, y_true, grad_true = x_true[0:1], y_true[0:1], [g[0:1] for g in grad_true]

# Apply defense
if args.defense == 'clip':
    grad_true = clip_gradients(grad_true, args.clip)
elif args.defense == 'noise':
    grad_true = add_noise(grad_true, args.sigma)

# Debug: Verificar estado dos gradientes
print("Requires grad status:", [p.requires_grad for p in model.parameters()])

# Run attack
# Run attack
start_time = time.time()
if args.attack == 'dlg':
    attack = DLGAttack(model, grad_true, lr=args.lr, optimizer_name=args.opt, num_iters=args.iters, num_restarts=args.restarts, device=device)
    (x_recon, y_recon), best_loss_val = attack.run(input_shape)  # Desempacotar a tupla
elif args.attack == 'idlg':
    attack = IDLGAttack(model, grad_true, lr=args.lr, optimizer_name=args.opt, num_iters=args.iters, num_restarts=args.restarts, device=device)
    (x_recon, y_recon), best_loss_val = attack.run(input_shape)  # Desempacotar a tupla
elapsed = time.time() - start_time

# E depois usar best_loss_val nas métricas
grad_loss = best_loss_val  # Já que best_loss_val é a perda de gradiente final

# Ajustar x_recon para o formato correto
if isinstance(x_recon, tuple):
    x_recon = x_recon[0]  # Pegar o primeiro elemento se for tupla
x_recon = x_recon.unsqueeze(0) if x_recon.dim() == 3 else x_recon  # Garantir formato (1, 1, 28, 28)

# Compute metrics
grad_loss = sum((g_r - g_t).pow(2).sum() for g_r, g_t in zip(torch.autograd.grad(model(x_recon).sum(), model.parameters(), create_graph=True), grad_true) if g_t is not None)
mse = torch.nn.functional.mse_loss(x_true, x_recon).item() if x_true.shape == x_recon.shape else 0.0
# Ajustar psnr para trabalhar com tensor ou float
mse_tensor = torch.tensor(mse, device=device) if isinstance(mse, float) else mse
psnr = 10 * torch.log10(torch.tensor(1.0, device=device) / (mse_tensor + 1e-8)).item() if mse_tensor > 0 else 0.0
ssim_val = 0.0  # Placeholder, implemente com skimage.metrics.ssim se necessário
best_loss = grad_loss.item() if grad_loss else 0.0

# Ajustar y_recon para lidar com float
# Ajustar y_recon para garantir que sejam inteiros
if isinstance(y_recon, (int, float)):
    y_recon = torch.tensor([int(y_recon)], device=device)
elif isinstance(y_recon, torch.Tensor):
    if y_recon.dtype != torch.long:  # Se não for do tipo longo, converter
        y_recon = y_recon.round().to(torch.long)  # Arredondar e converter para inteiro
    y_recon = y_recon.to(device)

# Verificar se o número de rótulos corresponde ao esperado
expected_labels = 1 if args.scenario == 'single' else 4
if len(y_recon) != expected_labels:
    print(f"Warning: Expected {expected_labels} labels, got {len(y_recon)}")
    # Se não corresponder, usar os primeiros 'expected_labels' ou preencher
    if len(y_recon) > expected_labels:
        y_recon = y_recon[:expected_labels]
    else:
        y_recon = torch.cat([y_recon, torch.zeros(expected_labels - len(y_recon), dtype=torch.long, device=device)])

# Save CSV
header = ["model", "attack", "defense", "opt", "scenario", "lr", "iters", "restarts", "clip", "sigma", "grad_loss", "mse", "psnr", "ssim", "time", "label_true", "label_recon"]
row = [args.model, args.attack, args.defense, args.opt, args.scenario, args.lr, args.iters, args.restarts, args.clip, args.sigma, best_loss, mse, psnr, ssim_val, elapsed, y_true.tolist(), y_recon.tolist()]

write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(row)

# Save reconstructed vs original image
save_image(torch.cat([x_true, x_recon], dim=0), fig_path, nrow=2, normalize=True, value_range=(0, 1))

print(f"\nResultados salvos em: {csv_path}")
print(f"Figura salva em: {fig_path}")