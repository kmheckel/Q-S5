# # full precision
# echo "smnist-full"
# sbatch run_smnist.sh \
#     --run_name=smnist-full-lnrm --batchnorm=False

# W8A8
echo "smnist-W8A8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W8A8-lnrm

### Towards 4-bit weights

# W8A8 for SSM, W4A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "smnist-W4A8Wssm8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W4A8Wssm8-lnrm
# W8A8 for A, W4A8 for everything else
echo "smnist-W4A8Wa8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W4A8Wa8-lnrm
# W4A8
echo "smnist-W4A8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W4A8-lnrm

### Towards 2-bit weights

# W8A8 for SSM, W2A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "smnist-W2A8Wssm8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W2A8Wssm8-lnrm
# W8A8 for A, W2A8 for everything else
echo "smnist-W2A8Wa8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W2A8Wa8-lnrm
# W2A8
echo "smnist-W2A8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=2 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W2A8-lnrm
