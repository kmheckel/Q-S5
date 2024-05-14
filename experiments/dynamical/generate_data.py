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

def generate(directory, **kwargs):
  directory = pathlib.Path(directory)
  if not directory.exists():
    directory.mkdir(parents=True)
    
  dataset = make_trajectory_ensemble(**kwargs)
  for k, v in dataset.items():
    filename = directory / k
    np.save(filename, v)

# %%
granularity = 100 # 100 = Fine
generate("data", n = 20000, resample=True, pts_per_period=granularity, random_state=1, use_tqdm=True)


