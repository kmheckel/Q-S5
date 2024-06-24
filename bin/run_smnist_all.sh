# # full precision
# echo "smnist-full"
# sbatch run_smnist.sh \
#     --run_name=smnist-full

# # W8
# # ---- is this needed? maybe only do this if W8A8 fails?
# sbatch run_smnist.sh \
#     --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8

# W8A8
echo "smnist-W8A8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W8A8

### Towards 4-bit weights

# W8A8 for SSM, W4A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "smnist-W4A8Wssm8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W4A8Wssm8
# W8A8 for A, W4A8 for everything else
echo "smnist-W4A8Wa8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W4A8Wa8
# W4A8
echo "smnist-W4A8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W4A8

### Towards 2-bit weights

# W8A8 for SSM, W2A8 for non-SSM
# ---- is this needed? we mainly care about recurrent vs. feedforward
echo "smnist-W2A8Wssm8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W2A8Wssm8
# W8A8 for A, W2A8 for everything else
echo "smnist-W2A8Wa8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W2A8Wa8
# W2A8
echo "smnist-W2A8"
sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=2 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --qgelu_approx --hard_sigmoid \
    --run_name=smnist-W2A8


# ######################################################
# ### Bonus experiments - quantizing activations further
# ######################################################


# ### Towards 4-bit activations

# # W8 everywhere, A4 non-SSM, A8 SSM
# sbatch run_smnist.sh --non_ssm_act_bits=4 --ssm_act_bits=8 \
#     --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8
# # W8A4
# sbatch run_smnist.sh --non_ssm_act_bits=4 --ssm_act_bits=4 \
#     --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8

# ### Towards 2-bit activations

# # W8 everywhere, A2 non-SSM, A8 SSM
# sbatch run_smnist.sh --non_ssm_act_bits=2 --ssm_act_bits=8 \
#     --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8
# # W8A2
# sbatch run_smnist.sh --non_ssm_act_bits=2 --ssm_act_bits=2 \
#     --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8

# ### Towards 1-bit activations

# sbatch run_smnist.sh --non_ssm_act_bits=8 --ssm_act_bits=8 --a_bits=8 --b_bits=1 --c_bits=1 --d_bits=1 --non_ssm_bits=1
# sbatch run_smnist.sh --non_ssm_act_bits=1 --ssm_act_bits=8 --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8
