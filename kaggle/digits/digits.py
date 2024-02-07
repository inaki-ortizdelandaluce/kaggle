import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from kaggle.digits.dataset import DigitRecognizerDataset
from kaggle.digits.models import DigitRecognizerModelV0
from kaggle.metrics import accuracy_fn
import os


def train_single_epoch(data_loader: torch.utils.data.DataLoader,
                       model: torch.nn.Module,
                       loss_fn: torch.nn.Module,
                       device: torch.device):
    model.train()

    train_loss, train_accuracy = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # forward pass
        y_pred = model(X)
        # calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        accuracy = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        train_accuracy += accuracy
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        # step optimizer
        optimizer.step()

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)

    return train_loss, train_accuracy


def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   device: torch.device):
    model.eval()
    y_eval = []
    with torch.inference_mode():
        for X in data_loader:
            y = model(X.to(device))
            y_eval.extend(y.argmax(dim=1).numpy())
    return y_eval


def write_submission(labels: list, output_dir: str):
    path = os.path.join(output_dir, "sample_submission.csv")
    results = pd.DataFrame({"ImageId": list(range(1, len(labels) + 1)), "Label": labels})
    results.to_csv(path, index=False, header=True)
    return None


def plot_sample(X, y):
    import matplotlib.pyplot as plt
    plt.imshow(X.squeeze(), cmap="gray")
    plt.title(f"{y}")
    plt.show()


if __name__ == "__main__":
    # hyperparameters
    epochs = 10
    learning_rate = 0.01
    batch_size = 32

    # set-up device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset and data loaders
    train_dataset = DigitRecognizerDataset(csv_file="../../data/digits/train.csv",
                                           train=True,
                                           transform=ToTensor())
    test_dataset = DigitRecognizerDataset(csv_file="../../data/digits/test.csv",
                                          train=False,
                                          transform=ToTensor())

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    # model
    model_0 = DigitRecognizerModelV0(input_shape=784,
                                     hidden_units=10,
                                     output_shape=len(train_dataset.classes)).to(device)

    # loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=learning_rate)

    # train model
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}\n------------")
        loss, acc = train_single_epoch(data_loader=train_dataloader, model=model_0, loss_fn=loss_fn, device=device)
        print(f"Train loss {loss:.5f} | Train accuracy : {acc:.2f}%")

    # evaluate model
    y_test = evaluate_model(model=model_0, data_loader=test_dataloader, device=device)

    # save results
    write_submission(labels=y_test, output_dir="../../data/digits")

