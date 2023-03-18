import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def train_single_epoch(model, optimizer, criterion, dataloader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    num_samples = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_acc += accuracy(output, target) * data.size(0)
        num_samples += data.size(0)

    train_loss /= num_samples
    train_acc /= num_samples

    return train_loss, train_acc



def eval_single_epoch(model, criterion, dataloader):
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            eval_loss += loss.item() * data.size(0)
            eval_acc += accuracy(output, target) * data.size(0)
            num_samples += data.size(0)

    eval_loss /= num_samples
    eval_acc /= num_samples

    return eval_loss, eval_acc


def train_model(config):
    
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    my_dataset = MyDataset(images_path='./data/data', labels_path='./chinese_mnist.csv', transform=trans)
    my_model = MyModel().to(device)

    optimizer = torch.optim.SGD(my_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=config["batch_size"], shuffle=False)

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(my_model, optimizer, criterion, train_loader)
        val_loss, val_acc = eval_single_epoch(my_model, criterion, val_loader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    save_model(my_model, "my_model.pth")
    return my_model


if __name__ == "__main__":

    config = {
        "batch_size": 64,
        "learning_rate": 0.1,
        "epochs": 5,
        "momentum": 0.9

    }
    train_model(config)
