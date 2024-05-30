# Check if the run name was provided
if [ -z "$1" ]; then
  echo "Error: No run name provided."
  echo "Usage: $0 run_name"
  exit 1
fi

# Assign the first command line argument to a variable
run_name="$1"
echo run_name

# # full precision - BN
# echo "listops-full"
# sbatch run_listops.sh \
#     --run_name="${run_name}-full-bn"

# # full precision - LN
# echo "listops-full-ln"
# sbatch run_listops.sh \
#     --batchnorm=False --run_name="${run_name}-full-ln"

# # full precision - LN without bias
# echo "listops-full-ln_nb"
# sbatch run_listops.sh \
#     --batchnorm=False --use_layernorm_bias=False \
#     --run_name="${run_name}-full-ln_nb"

# W8A8
echo "listops-W8A8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W8A8"

### Towards 4-bit weights

# W8A8 for SSM, W4A8 for non-SSM
echo "listops-W4A8Wssm8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W4A8Wssm8-lnrm"
# W8A8 for A, W4A8 for everything else
echo "listops-W4A8Wa8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W4A8Wa8-lnrm"
# W4A8
echo "listops-W4A8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W4A8-lnrm"

### Towards 2-bit weights

# W8A8 for SSM, W2A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "listops-W2A8Wssm8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W2A8Wssm8-lnrm"
# W8A8 for A, W2A8 for everything else
echo "listops-W2A8Wa8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W2A8Wa8-lnrm"
# W2A8
echo "listops-W2A8"
sbatch run_listops.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=2 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name="${run_name}-W2A8-lnrm"