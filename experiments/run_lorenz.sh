#!/bin/env bash

# Run the Lorenz system experiment for different bits
A_BITS="1 2 4 8 16"
B_BITS="1 2 4 8 16"
C_BITS="1 2 4 8 16"
D_BITS="1 2 4 8 16"
SSM_ACT_BITS="1 2 4 8 16"
#NON_SSM_BITS="1 2 5 8 16"
#NON_SSM_ACT_BITS="1 2 4 8 16"

for a_bits in $A_BITS; do
   for ssm_act_bits in $SSM_ACT_BITS; do
      echo "Running Lorentz system with a_bits=$a_bits, b_bits=$b_bits, c_bits=$c_bits, d_bits=$d_bits, ssm_act_bits=$ssm_act_bits"
      #python3 run_experiment.py --a_bits $a_bits --b_bits $b_bits --c_bits $c_bits --d_bits $d_bits --ssm_act_bits $ssm_act_bits --experiment lorenz
      python3 run_experiment.py --a_bits $a_bits --ssm_act_bits $ssm_act_bits --experiment lorenz
    done
done
