# echo "W8A8"
# sbatch qaa_smnist.sh \
#     --load_run_name=smnist-full-ln_nb --batchnorm=False \
#     --non_ssm_act_bits=8 --ssm_act_bits=8 \
#     --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
#     --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
#     --run_name=smnist-qaa-W8A8

# towards 4-bit weights

echo "W4A8Wssm8"
sbatch qaa_smnist.sh \
    --load_run_name=smnist-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
    --run_name=smnist-qaa-W4A8Wssm8

echo "W4A8Wa8"
sbatch qaa_smnist.sh \
    --load_run_name=smnist-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
    --run_name=smnist-qaa-W4A8Wa8

echo "W4A8"
sbatch qaa_smnist.sh \
    --load_run_name=smnist-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
    --run_name=smnist-qaa-W4A8

# towards 2-bit weights

echo "W2A8Wssm8"
sbatch qaa_smnist.sh \
    --load_run_name=smnist-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
    --run_name=smnist-qaa-W2A8Wssm8

echo "W2A8Wa8"
sbatch qaa_smnist.sh \
    --load_run_name=smnist-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
    --run_name=smnist-qaa-W2A8Wa8

echo "W2A8"
sbatch qaa_smnist.sh \
    --load_run_name=smnist-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=2 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --use_qlayernorm_if_quantized=True --hard_sigmoid --qgelu_approx \
    --run_name=smnist-qaa-W2A8