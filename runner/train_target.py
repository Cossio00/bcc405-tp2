import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.mlp import MLP
from models.cnn_small import SmallCNN

def get_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

def train_model(model_name="mlp", epochs=5, device="cpu"):
    if model_name == "mlp":
        model = MLP()
    else:
        model = SmallCNN()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader()

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model
