from pathlib import Path
from typing import Tuple, Dict

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

    def __init__(self, path: str, timesteps: int, train: bool):
        self.data = jnp.load(path)
        self.timesteps = timesteps
        self.indices = jnp.arange(len(self.data) // self.timesteps)
        fraction = int(len(self.indices) * 0.8)
        self.indices = self.indices[:fraction] if train else self.indices[fraction:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_index = idx * self.timesteps
        end_index = start_index + self.timesteps
        return self.data[start_index:end_index]


def dysts_collate_fn(batch):
    stacked = jnp.stack(batch)
    return stacked[:, :-1], stacked[:, 1:]


def dysts_data_loader(path, timesteps, bsz, train=True):
    d = DystsDataset(path, timesteps, train=train)
    return torch.utils.data.DataLoader(
        d, batch_size=bsz, shuffle=True, collate_fn=dysts_collate_fn
    )


def lorenz_data_fn(path, timesteps, seed, bsz) -> ReturnType:
    trainset = dysts_data_loader(path, timesteps, bsz, train=True)
    valset = dysts_data_loader(path, timesteps, bsz, train=False)
    testset = None
    seq_len = 128
    in_dim = 3
    return trainset, valset, testset, {}, None, seq_len, in_dim, len(trainset)