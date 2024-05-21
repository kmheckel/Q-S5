from dysts.flows import MackeyGlass
import numpy as np
import pathlib
import ray
import tqdm

ray.init()

def make_gen_trajectory(**kwargs):
  @ray.remote
  def gen_trajectory(tau):
    MG = MackeyGlass(tau=tau)
    MG.ic = np.random.random(MG.ic.shape)
    return MG.make_trajectory(**kwargs)

  return gen_trajectory

def generate(directory, runs, max_tau=30, stride=1, **kwargs):
  directory = pathlib.Path(directory)
  if not directory.exists():
    directory.mkdir(parents=True)

  gen_mg = make_gen_trajectory(**kwargs)

  for tau in tqdm.trange(1, max_tau+1, stride):
    sampled_trajectory_futures = [gen_mg.remote(tau) for i in range(runs)]
    sampled_trajectories = ray.get(sampled_trajectory_futures)
    if not (directory / f"MackeyGlass").exists():
      (directory / f"MackeyGlass").mkdir(parents=True)
    (directory / f"MackeyGlass" / f"tau_{tau}").mkdir(parents=True)
    for i, traj in enumerate(sampled_trajectories):
      filename = directory / f"MackeyGlass" / f"tau_{tau}" / f"{i}"
      np.save(filename, traj)

generate("data", runs=16, n = 1024, resample=True, pts_per_period=100)