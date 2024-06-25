#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi
dataset_name=$1

# non_ssm_bits, a_bits, bcd_bits, run_name_suffix
declare -a configurations=(
    "8 8 8 W8A8"
    "4 8 8 W4A8Wssm8"
    "4 8 4 W4A8Wa8"
    "4 4 4 W4A8"
    "2 8 8 W2A8Wssm8"
    "2 8 2 W2A8Wa8"
    "2 2 2 W2A8"
)
for config in "${configurations[@]}"; do
    IFS=' ' read -r -a params <<< "$config"
    # Extract parameters
    non_ssm_bits=${params[0]}
    a_bits=${params[1]}
    bcd_bits=${params[2]}
    run_name_suffix=${params[3]}
    # log run config and then run
    echo "${run_name_suffix}"
    sbatch qaa_${dataset_name}.sh \
        --load_run_name=${dataset_name}-full-ln_nb --batchnorm=False \
        --non_ssm_act_bits=8 --ssm_act_bits=8 --non_ssm_bits=$non_ssm_bits \
        --a_bits=$a_bits --b_bits=$bcd_bits --c_bits=$bcd_bits --d_bits=$bcd_bits \
        --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
        --run_name=${dataset_name}-qaa-${run_name_suffix}
done