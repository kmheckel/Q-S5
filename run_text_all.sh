# # full precision - BN
# echo "text-full"
# sbatch run_text.sh \
#     --run_name=text-full-bn

# full precision - LN
echo "text-full"
sbatch run_text.sh \
    --batchnorm=False --run_name=text-full-ln

# full precision - LN without bias
echo "text-full-ln_nb"
sbatch run_text.sh \
    --batchnorm=False --use_layernorm_bias=False \
    --run_name=text-full-ln_nb

# W8A8
echo "text-W8A8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W8A8

### Towards 4-bit weights

# W8A8 for SSM, W4A8 for non-SSM
echo "text-W4A8Wssm8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W4A8Wssm8-lnrm
# W8A8 for A, W4A8 for everything else
echo "text-W4A8Wa8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W4A8Wa8-lnrm
# W4A8
echo "text-W4A8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W4A8-lnrm

### Towards 2-bit weights

# W8A8 for SSM, W2A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "text-W2A8Wssm8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W2A8Wssm8-lnrm
# W8A8 for A, W2A8 for everything else
echo "text-W2A8Wa8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W2A8Wa8-lnrm
# W2A8
echo "text-W2A8"
sbatch run_text.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=2 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=text-W2A8-lnrm