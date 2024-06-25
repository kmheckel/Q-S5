# W8A8
echo "scifar-W8A8"
sbatch run_scifar.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=scifar-W8A8

### Towards 4-bit weights

# W8A8 for SSM, W4A8 for non-SSM
echo "scifar-W4A8Wssm8"
sbatch run_scifar.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=scifar-W4A8Wssm8-lnrm
# W8A8 for A, W4A8 for everything else
echo "scifar-W4A8Wa8"
sbatch run_scifar.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=scifar-W4A8Wa8-lnrm

### Towards 2-bit weights

# W8A8 for SSM, W2A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "scifar-W2A8Wssm8"
sbatch run_scifar.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=scifar-W2A8Wssm8-lnrm