echo "FP: eval-ptq--scifar-full-ln_nb--fp"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --use_layernorm_bias=False \
    --run_name=eval-ptq--scifar-full-ln_nb--fp

echo "W8A8"
echo "W8A8 gelu, sigmoid, LayerNorm: eval-ptq--scifar-full-ln_nb--W8A8-gelu-sigmoid-ln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=False --use_layernorm_bias=False \
    --run_name=eval-ptq--scifar-full-ln_nb--W8A8-gelu-sigmoid-ln

echo "W8A8 + qgelu"
echo "W8A8 qgelu, sigmoid, LayerNorm: eval-ptq--scifar-full-ln_nb--W8A8-qgelu-sigmoid-ln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=False --use_layernorm_bias=False \
    --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W8A8-qgelu-sigmoid-ln

echo "W8A8 + hard sigmoid"
echo "W8A8 gelu, hard sigmoid, LayerNorm: eval-ptq--scifar-full-ln_nb--W8A8-gelu-hsigmoid-ln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=False --use_layernorm_bias=False \
    --hard_sigmoid \
    --run_name=eval-ptq--scifar-full-ln_nb--W8A8-gelu-hsigmoid-ln

echo "W8A8 + qlayernorm"
echo "W8A8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W8A8-gelu-sigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --run_name=eval-ptq--scifar-full-ln_nb--W8A8-gelu-sigmoid-qln

echo "W8A8 + qgelu + hard sigmoid + qlayernorm"
echo "W8A8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W8A8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W8A8-qgelu-hsigmoid-qln


#### Lower quantizations


echo "W4A8Wssm8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W4A8Wssm8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=4 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W4A8Wssm8-qgelu-hsigmoid-qln

echo "W4A8Wa8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W4A8Wa8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W4A8Wa8-qgelu-hsigmoid-qln

echo "W4A8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W4A8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=4 --b_bits=4 --c_bits=4 --d_bits=4 --non_ssm_bits=4 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W4A8-qgelu-hsigmoid-qln

echo "W2A8Wssm8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W2A8Wssm8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=2 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W2A8Wssm8-qgelu-hsigmoid-qln

echo "W2A8Wa8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W2A8Wa8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=8 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W2A8Wa8-qgelu-hsigmoid-qln

echo "W2A8 Gelu, Sigmoid, qLayerNorm: eval-ptq--scifar-full-ln_nb--W2A8-qgelu-hsigmoid-qln"
sbatch eval_scifar.sh \
    --load_run_name=scifar-full-ln_nb --batchnorm=False \
    --non_ssm_act_bits=8 --ssm_act_bits=8 \
    --a_bits=2 --b_bits=2 --c_bits=2 --d_bits=2 --non_ssm_bits=2 \
    --use_qlayernorm_if_quantized=True --use_layernorm_bias=True \
    --hard_sigmoid --qgelu_approx \
    --run_name=eval-ptq--scifar-full-ln_nb--W2A8-qgelu-hsigmoid-qln
