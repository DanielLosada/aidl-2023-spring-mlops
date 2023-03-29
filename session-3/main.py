import torch
import os
from torch.utils.data import DataLoader

from model import MyModel
from utils import binary_accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer, criterion):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        # You will need to do y = y.unsqueeze(1).float() to add an output dimension to the labels and cast to the correct type
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        acc = binary_accuracy(y, output)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader, criterion):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            y = y.unsqueeze(1).float()
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            acc = binary_accuracy(y, output)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):
    print("Hola: ", os.getcwd())
    data_transforms = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                                    std=[0.5])
                                                    ])
    train_dataset = ImageFolder("./session-3/dataset/cars_vs_flowers/training_set", data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = ImageFolder("./session-3/dataset/cars_vs_flowers/test_set", data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel().to(device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer, criterion)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, test_loader, criterion)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    return my_model


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 20,
    }
    my_model = train_model(config)

    
