import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import os


class DigitRecognizerDataset(Dataset):
    def __init__(self,
                 csv_file,
                 transform,
                 train=True):
        super().__init__()
        self.train = train
        self.transform = transform
        self.classes = None
        if self.train:
            self.data_frame = pd.read_csv(csv_file, header="infer")
            self.classes = self.data_frame['label'].unique()
        else:
            self.data_frame = pd.read_csv(csv_file, header="infer")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.train:
            flattened = self.data_frame.iloc[idx, 1:].values.astype(np.uint8)
            # reshape image allowing for further conversion using PyTorch's transform ToTensor(),
            # which converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
            image = self.transform(np.reshape(flattened, (28, 28, 1)))
            label = self.data_frame.iloc[idx, 0]
            return image, label
        else:
            flattened = self.data_frame.iloc[idx, :].values.astype(np.uint8)
            image = self.transform(np.reshape(flattened, (28, 28, 1)))
            return image


class DigitRecognizerModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            # nn.ReLU()
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


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


if __name__ == "__main__":
    # hyperparameters
    epochs = 10
    learning_rate = 0.01
    batch_size = 32

    # set-up device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset and data loaders
    train_dataset = DigitRecognizerDataset(csv_file="/kaggle/input/digit-recognizer/train.csv",
                                           train=True,
                                           transform=ToTensor())
    test_dataset = DigitRecognizerDataset(csv_file="/kaggle/input/digit-recognizer/test.csv",
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
    write_submission(labels=y_test, output_dir="/kaggle/working")
