## make sure you are using python 3.10.10
# pyenv global 3.10.10
python -m venv venv
source venv/bin/activate

pip install -U pip
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

cd aqt
pip install .
cd ..

pip install wandb tqdm einops
pip install torch torchtext torchaudio torchvision
pip install datasets tensorflow-datasets
