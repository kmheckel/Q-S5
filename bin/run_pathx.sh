#!/bin/bash
#SBATCH -p g80
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 24:00:00
#SBATCH --qos=high

source ./venv/bin/activate
cd /home/apierro/NeuroSSMs/S5fork

python run_train.py \
    --run_name=pathx-fp16 --checkpoint_dir=/home/apierro/NeuroSSMs/final \
    --mlflow_tracking_uri="http://isl-cpu1.rr.intel.com:2517/" --mlflow_experiment_id=887581300639011956 \
    --mlflow_run_id=e80de4afee8d4615a5db779f69bab3ea \
    --job_id=$SLURM_JOB_ID \
    --C_init=complex_normal --batchnorm=True --bidirectional=True \
    --blocks=16 --bn_momentum=0.9 --bsz=32 --d_model=128 --dataset=pathx-classification \
    --dt_min=0.0001 --epochs=75 --jax_seed=6429262 --lr_factor=3 --n_layers=6 \
    --opt_config=BandCdecay --p_dropout=0.0 --ssm_lr_base=0.0006 --ssm_size_base=256 \
    --warmup_end=1 --weight_decay=0.06
