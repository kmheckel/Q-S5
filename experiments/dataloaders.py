from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import jax.numpy as jnp

from dysts.datasets import load_json

ReturnType = Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Dict,
    int,
    int,
    int,
    int,
]


def list_datasets(path: str) -> list:
    return [p for p in Path(path).glob("*.npy")]


class DystsDataset(torch.utils.data.Dataset):

    def __init__(
        self, path: str, timesteps: int, train: bool, forecast: Optional[int] = None
    ):
        """
        Args:
            path (str): Path to the dataset
            timesteps (int): Number of timesteps to consider
            train (bool): Whether to use the training or validation set
            forecast (Optional[int]): Number of timesteps to forecast, defaults to timesteps
        """
        self.data = jnp.load(path)
        self.timesteps = timesteps
        self.forecast = forecast if forecast is not None else timesteps
        self.indices = jnp.arange(len(self.data) // (self.timesteps + self.forecast))
        fraction = int(len(self.indices) * 0.8)
        self.indices = self.indices[:fraction] if train else self.indices[fraction:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_index = idx * self.timesteps
        mid_index = start_index + self.timesteps
        end_index = mid_index + self.forecast
        return self.data[start_index:mid_index], self.data[mid_index:end_index]


def dysts_collate_fn(batch):
    xs = jnp.stack([x for x, _ in batch])
    ys = jnp.stack([y for _, y in batch])
    return xs, ys


def dysts_data_loader(path, timesteps, bsz, train=True):
    d = DystsDataset(path, timesteps, train=train)
    return torch.utils.data.DataLoader(
        d, batch_size=bsz, shuffle=True, collate_fn=dysts_collate_fn
    )


def dysts_data_fn(path, timesteps, seed, bsz) -> ReturnType:
    trainset = dysts_data_loader(path, timesteps, bsz, train=True)
    valset = dysts_data_loader(path, timesteps, bsz, train=False)
    testset = None
    seq_len = 128
    in_dim = 3
    return trainset, valset, testset, {}, None, seq_len, in_dim, len(trainset)
