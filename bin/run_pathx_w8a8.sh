#!/bin/bash
#SBATCH -p g80
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 24:00:00
#SBATCH --qos=high

source ./venv/bin/activate
cd /home/apierro/NeuroSSMs/S5fork


python run_qtrain.py \
    --run_name=pathx-w8a8 --checkpoint_dir=/home/apierro/NeuroSSMs/final \
    --mlflow_tracking_uri="http://isl-cpu1.rr.intel.com:2517/" --mlflow_experiment_id=887581300639011956 \
    --mlflow_run_id=60cf96de4b57409093e20d2274e65e0b \
    --job_id=$SLURM_JOB_ID \
    --ssm_act_bits=8 --non_ssm_act_bits=8 \
    --non_ssm_bits=8 --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 \
    --C_init=complex_normal --batchnorm=True --bidirectional=True \
    --blocks=16 --bn_momentum=0.9 --bsz=32 --d_model=128 --dataset=pathx-classification \
    --dt_min=0.0001 --epochs=75 --jax_seed=642262 --lr_factor=3 --n_layers=6 \
    --opt_config=BandCdecay --p_dropout=0.0 --ssm_lr_base=0.0006 --ssm_size_base=256 \
    --warmup_end=1 --weight_decay=0.06
