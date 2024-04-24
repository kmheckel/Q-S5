from pathlib import Path
from typing import Tuple, Dict

import torch
import jax.numpy as jnp

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


class LorenzDataset(torch.utils.data.Dataset):

    def __init__(self, path: str, train: bool = True):
        self.files = list(Path(path).glob("*.npy"))
        if train:
            self.files = self.files[: int(0.8 * len(self.files))]
        else:
            self.files = self.files[int(0.8 * len(self.files)) :]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return jnp.load(self.files[idx])


def lorenz_collate_fn(batch):
    stacked = jnp.stack(batch)
    return stacked[:, :-1], stacked[:, 1:]


def lorenz_data_loader(path, bsz, train=True):
    d = LorenzDataset(path, train=train)
    return torch.utils.data.DataLoader(
        d, batch_size=bsz, shuffle=True, collate_fn=lorenz_collate_fn
    )


def lorenz_data_fn(path, seed, bsz) -> ReturnType:
    trainset = lorenz_data_loader(path, bsz, train=True)
    valset = lorenz_data_loader(path, bsz, train=False)
    testset = None
    seq_len = 128
    in_dim = 3
    return trainset, valset, testset, {}, None, seq_len, in_dim, len(trainset)
