# %% [markdown]
# # Generate dynamical systems dataset

# %%
from dysts.base import make_trajectory_ensemble
from dysts.datasets import convert_json_to_gzip

# %%
import tempfile
import json
import pathlib
import numpy as np
import tqdm

def generate(directory, runs, **kwargs):
  directory = pathlib.Path(directory)
  if not directory.exists():
    directory.mkdir(parents=True)
    
  for i in tqdm.trange(runs):
    dataset = make_trajectory_ensemble(random_state=i, **kwargs)
    for k, v in dataset.items():
      filename = directory / f"{i}_{k}"
      np.save(filename, v)

# %%
# Generate Lorenz
granularity = 100 # 100 = Fine
generate("data", runs = 1024, n = 1024, resample=True, pts_per_period=granularity, subset=["Lorenz"])

# %%
# Generate all
#granularity = 100 # 100 = Fine
#generate("data", runs = 1024, n = 1024, resample=True, pts_per_period=granularity, use_tqdm=True, subset="Lorenz")
