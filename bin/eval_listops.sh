#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 10:00:00
#SBATCH -o /home/sabreu/NeuroSSMs/logs/slurm-%j.out

# Default values
a_bits=None
ssm_act_bits=None
non_ssm_act_bits=None
non_ssm_bits=None
b_bits=None
c_bits=None
d_bits=None
qgelu_approx="False"
hard_sigmoid="False"
batchnorm="True"
run_name=None
load_run_name=None
checkpoint_dir="${HOME}/NeuroSSMs/checkpoints"
use_qlayernorm_if_quantized="True"
remove_norm_bias_from_checkpoint="False"
use_layernorm_bias="True"

# Parse arguments
for i in "$@"
do
case $i in
    --a_bits=*)
    a_bits="${i#*=}"
    shift # past argument=value
    ;;
    --ssm_act_bits=*)
    ssm_act_bits="${i#*=}"
    shift # past argument=value
    ;;
    --non_ssm_act_bits=*)
    non_ssm_act_bits="${i#*=}"
    shift # past argument=value
    ;;
    --non_ssm_bits=*)
    non_ssm_bits="${i#*=}"
    shift # past argument=value
    ;;
    --b_bits=*)
    b_bits="${i#*=}"
    shift # past argument=value
    ;;
    --c_bits=*)
    c_bits="${i#*=}"
    shift # past argument=value
    ;;
    --d_bits=*)
    d_bits="${i#*=}"
    shift # past argument=value
    ;;
    --hard_sigmoid)
    hard_sigmoid="True"
    shift # past argument with no value
    ;;
    --qgelu_approx)
    qgelu_approx="True"
    shift # past argument with no value
    ;;
    --batchnorm=*)
    batchnorm="${i#*=}"
    shift # past argument=value
    ;;
    --use_qlayernorm_if_quantized=*)
    use_qlayernorm_if_quantized="${i#*=}"
    shift # past argument=value
    ;;
    --remove_norm_bias_from_checkpoint=*)
    remove_norm_bias_from_checkpoint="${i#*=}"
    shift # past argument=value
    ;;
    --use_layernorm_bias=*)
    use_layernorm_bias="${i#*=}"
    shift # past argument=value
    ;;
    --run_name=*)
    run_name="${i#*=}"
    shift # past argument=value
    ;;
    --load_run_name=*)
    load_run_name="${i#*=}"
    shift # past argument=value
    ;;
    --checkpoint_dir=*)
    checkpoint_dir="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Prepare optional parameter strings
args=""
for var in remove_norm_bias_from_checkpoint use_layernorm_bias a_bits ssm_act_bits non_ssm_act_bits non_ssm_bits b_bits c_bits d_bits hard_sigmoid qgelu_approx use_qlayernorm_if_quantized batchnorm load_run_name run_name checkpoint_dir; do
    value=$(eval echo \$$var)
    if [ "$value" != "None" ]; then
        args+=" --$var=$value"
    fi
done

echo "Running with args: $args"

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
# NOTE: batchnorm is included in the $args variable! specify this from `sbatch run_smnist.sh --batchnorm={True|False}`
python run_qeval.py \
    --USE_WANDB=TRUE --wandb_project=NeuroSSMs --wandb_entity=il-ncl \
    $args \
    --dataset=listops-classification \
    --n_layers=8 --d_model=128 --ssm_size_base=16 --blocks=8 \
    --prenorm=True --bsz=50 --epochs=80 \
    --ssm_lr_base=0.001 --lr_factor=3 --p_dropout=0 --weight_decay=0.04 \
    --jax_seed=42 \
    --C_init=lecun_normal --bidirectional=True --clip_eigs=True \
    --opt_config=BfastandCdecay \
    --warmup_end=10 \
    --activation_fn=half_glu2
