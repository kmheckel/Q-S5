#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 24:00:00
#SBATCH --qos=high

source ./venv/bin/activate

cd /home/apierro/NeuroSSMs/S5fork

python run_train.py \
    --run_name=pathfinder-fp16 --checkpoint_dir=/home/apierro/NeuroSSMs/final \
    --mlflow_tracking_uri="http://isl-cpu1.rr.intel.com:2517/" --mlflow_experiment_id=458051151596686345 \
    --mlflow_run_id=c2bb3b8fade84451ac126cd538fc0abd \
    --job_id=$SLURM_JOB_ID \
    --C_init=trunc_standard_normal --batchnorm=True --bidirectional=True \
    --blocks=8 --bn_momentum=0.9 --bsz=64 --d_model=192 \
    --dataset=pathfinder-classification  --epochs=200 --jax_seed=8180844 --lr_factor=5 \
    --n_layers=6 --opt_config=standard --p_dropout=0.05 --ssm_lr_base=0.0009 \
    --ssm_size_base=256 --warmup_end=1 --weight_decay=0.03
