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
    return [p for p in Path(path).children() if p.is_dir()]


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
        if isinstance(path, str):
            path = Path(path)
        self.files = list(path.glob("**/*.npy"))
        self.files = sorted(self.files)
        self.timesteps = timesteps
        fraction = int(len(self.files) * 0.8)
        self.forecast = forecast if forecast is not None else timesteps
        self.files = self.files[:fraction] if train else self.files[fraction:]
        n_steps = jnp.load(self.files[0]).shape[0]
        if n_steps < self.timesteps + self.forecast:
            raise ValueError(f"Number of data timesteps ({n_steps}) is < timesteps + forecast")
        
        # self.data = jnp.stack([jnp.load(f) for f in self.files])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = jnp.load(self.files[idx])
        mid_index = self.timesteps
        end_index = mid_index + self.forecast
        return data[:mid_index], data[mid_index:end_index]


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
    testset = dysts_data_loader(path, timesteps, bsz, train=False)
    valset = None
    seq_len = timesteps
    in_dim = trainset.dataset[0][0].shape[-1]
    return trainset, valset, testset, {}, None, seq_len, in_dim, len(trainset)
