import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.mlp import MLP
from models.cnn_small import SmallCNN

def get_dataloader(batch_size=32, scenario="single"):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    batch_size = 1 if scenario == 'single' else 4  # Ajustar batch_size com base em scenario
    return torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

def train_model(model_name="mlp", epochs=25, device="cpu", scenario="single"):
    if model_name == "mlp":
        model = MLP()
    else:
        model = SmallCNN()
    model = model.to(device)
    model.train()  # Garantir modo de treinamento
    for param in model.parameters():
        param.requires_grad_(True)  # Habilitar gradientes

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(scenario=scenario)  # Passar scenario para get_dataloader

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Calcular e salvar gradientes fora de torch.no_grad()
    os.makedirs("results/grads", exist_ok=True)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)  # Calcular gradientes
        torch.save({
            'model_state': model.state_dict(),
            'grads': [g.detach() for g in grads],  # Detach para evitar referência ao grafo
            'data': (x.detach(), y.detach())
        }, f'results/grads/{model_name}_grads_{scenario}.pt')  # Incluir scenario no nome
        break  # Salvar apenas um exemplo
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train target model and save gradients")
    parser.add_argument("--model", type=str, required=True, choices=['mlp', 'cnn'])
    parser.add_argument("--scenario", type=str, required=True, choices=['single', 'batch'])
    args = parser.parse_args()

    device = torch.device("cpu")  # Ajustar para 'cuda' se disponível
    train_model(args.model, device=device, scenario=args.scenario)