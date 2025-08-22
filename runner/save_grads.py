import os
import torch
from torchvision import datasets, transforms
from models.mlp import MLP
from models.cnn_small import SmallCNN

os.makedirs("results/grads", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

def save_grads(model_name="mlp", num_samples=4):
    if model_name == "mlp":
        model = MLP().to(device)
    else:
        model = SmallCNN().to(device)

    # dados
    x = torch.stack([dataset[i][0] for i in range(num_samples)], dim=0).to(device)
    y = torch.tensor([dataset[i][1] for i in range(num_samples)], device=device)

    # forward + backward
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    out = model(x.view(x.size(0), -1) if model_name=="mlp" else x)
    loss = criterion(out, y)
    loss.backward()

    grads = [p.grad.clone().detach() for p in model.parameters()]

    ckpt = {
        "model_state": model.state_dict(),
        "grads": grads,
        "data": (x.cpu(), y.cpu())
    }
    torch.save(ckpt, f"results/grads/{model_name}_grads.pt")
    print(f"Gradientes salvos em results/grads/{model_name}_grads.pt")

if __name__ == "__main__":
    save_grads("mlp", num_samples=4)
    save_grads("cnn", num_samples=4)
