### Towards 4-bit activations

# W8 everywhere, A4 non-SSM, A8 SSM
echo "smnist-W8A4Assm8-lnrm"
sbatch run_smnist.sh --non_ssm_act_bits=4 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W8A4Assm8-lnrm

# W8A4
echo "smnist-W8A4-lnrm"
sbatch run_smnist.sh --non_ssm_act_bits=4 --ssm_act_bits=4 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W8A4-lnrm

### Towards 2-bit activations

# W8 everywhere, A2 non-SSM, A8 SSM
echo "smnist-W8A2Assm8-lnrm"
sbatch run_smnist.sh --non_ssm_act_bits=2 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W8A2Assm8-lnrm

# W8A2
echo "smnist-W8A2-lnrm"
sbatch run_smnist.sh --non_ssm_act_bits=2 --ssm_act_bits=2 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid --batchnorm=False \
    --run_name=smnist-W8A2-lnrm
