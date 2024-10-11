import torch
from torch.utils.data import Dataset


class TSDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).to(torch.float32)
        self.target = torch.from_numpy(target).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    