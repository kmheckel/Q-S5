#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -t 10:00:00

# NOTE: for quantized runs take ~6x as long as non-quantized

# setup pyenv
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

###############################################################

export WANDB_APIKEY="$(cat wandb_apikey.txt)"

# activate venv
source ./venv/bin/activate

# enter the right directory
cd /home/sabreu/NeuroSSMs/S5fork

# run the script
# d_model           H: dims for input/output features
# ssm_size_base     P: latent size in SSM
# blocks            J: number of blocks used to initialize A
python run_qtrain.py \
    --USE_WANDB=TRUE --wandb_project=qSSMs --wandb_entity=stevenabreu7 --wandb_apikey="$WANDB_APIKEY" \
    --dataset=mnist-classification \
    --a_bits=8 \
    --ssm_act_bits=8 --non_ssm_act_bits=8 \
    --non_ssm_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 \
    --n_layers=4 --d_model=96 --ssm_size_base=128 --blocks=1 \
    --batchnorm=TRUE --prenorm=TRUE --bidirectional=FALSE \
    --ssm_lr_base=0.004 --lr_factor=4 --p_dropout=0.1 --weight_decay=0.01 \
    --bsz=50 --epochs=150
