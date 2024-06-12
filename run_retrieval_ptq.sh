#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 24:00:00
#SBATCH --qos=high

source ./venv/bin/activate
cd /home/apierro/NeuroSSMs/S5fork

python run_qtrain.py \
    --run_name=retrieval-fp16-ptq --checkpoint_dir=/home/apierro/NeuroSSMs/final \
    --mlflow_tracking_uri="http://isl-cpu1.rr.intel.com:2517/" --mlflow_experiment_id=676608297636244909 \
    --job_id=$SLURM_JOB_ID \
    --C_init=trunc_standard_normal --batchnorm=False --bidirectional=True --use_layernorm_bias=False \
    --blocks=16 --bsz=32 --d_model=128 --dataset=aan-classification \
    --dt_global=True --epochs=20 --jax_seed=5464368 --lr_factor=2 --n_layers=6 \
    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.001 --ssm_size_base=256 \
    --warmup_end=1 --weight_decay=0.05