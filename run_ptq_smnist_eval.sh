echo "FP: eval-ptq--smnist-full-lnrm--fp"
sbatch eval_smnist.sh \
    --load_run_name=smnist-full-lnrm --batchnorm=False \
    --run_name=eval-ptq--smnist-full-lnrm--fp

echo "W8A8"
echo "W8A8 gelu, sigmoid, LayerNorm: eval-ptq--smnist-full-lnrm--W8A8-gelu-sigmoid-ln"
sbatch eval_smnist.sh \
    --load_run_name=smnist-full-lnrm --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=False \
    --run_name=eval-ptq--smnist-full-lnrm--W8A8-gelu-sigmoid-ln

echo "W8A8 + qgelu"
echo "W8A8 qgelu, sigmoid, LayerNorm: eval-ptq--smnist-full-lnrm--W8A8-qgelu-sigmoid-ln"
sbatch eval_smnist.sh \
    --load_run_name=smnist-full-lnrm --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=False \
    --qgelu_approx \
    --run_name=eval-ptq--smnist-full-lnrm--W8A8-qgelu-sigmoid-ln

echo "W8A8 + hard sigmoid"
echo "W8A8 gelu, hard sigmoid, LayerNorm: eval-ptq--smnist-full-lnrm--W8A8-gelu-hsigmoid-ln"
sbatch eval_smnist.sh \
    --load_run_name=smnist-full-lnrm --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=False \
    --hard_sigmoid \
    --run_name=eval-ptq--smnist-full-lnrm--W8A8-gelu-hsigmoid-ln

echo "W8A8 + qlayernorm"
echo "W8A8 Gelu, Sigmoid, qLayerNorm: eval-ptq--smnist-full-lnrm--W8A8-gelu-sigmoid-qln"
sbatch eval_smnist.sh \
    --load_run_name=smnist-full-lnrm --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=True --remove_norm_bias_from_checkpoint=True \
    --run_name=eval-ptq--smnist-full-lnrm--W8A8-gelu-sigmoid-qln

echo "W8A8 + qgelu + hard sigmoid + qlayernorm"
echo "W8A8 Gelu, Sigmoid, qLayerNorm: eval-ptq--smnist-full-lnrm--W8A8-qgelu-hsigmoid-qln"
sbatch eval_smnist.sh \
    --load_run_name=smnist-full-lnrm --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=True --remove_norm_bias_from_checkpoint=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--smnist-full-lnrm--W8A8-qgelu-hsigmoid-qln
