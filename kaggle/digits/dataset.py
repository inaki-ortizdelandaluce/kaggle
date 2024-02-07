from torch.utils.data import Dataset
import pandas as pd
import numpy as np


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
