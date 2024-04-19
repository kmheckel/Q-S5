from pathlib import Path

import torch
import jax.numpy as jnp


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
