#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 10:00:00
#SBATCH -o /home/sabreu/NeuroSSMs/logs/slurm-%j.out

# NOTE: for quantized runs take ~6x as long as non-quantized

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
checkpoint_dir="${HOME}/NeuroSSMs/checkpoints"

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
    --run_name=*)
    run_name="${i#*=}"
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
for var in a_bits ssm_act_bits non_ssm_act_bits non_ssm_bits b_bits c_bits d_bits hard_sigmoid qgelu_approx batchnorm run_name checkpoint_dir; do
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
python run_qtrain.py \
    --USE_WANDB=TRUE --wandb_project=qSSMs --wandb_entity=stevenabreu7 \
    $args \
    --dataset=mnist-classification \
    --n_layers=4 --d_model=96 --ssm_size_base=128 --blocks=1 \
    --prenorm=TRUE --bidirectional=FALSE \
    --ssm_lr_base=0.004 --lr_factor=4 --p_dropout=0.1 --weight_decay=0.01 \
    --bsz=50 --epochs=150
