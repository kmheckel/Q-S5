#!/bin/env bash

# Run the MackeyGlass system experiment for different bits
A_BITS="1 2 3 4 5 6 7 8 16 None"
B_BITS="1 2 4 8 16"
C_BITS="1 2 4 8 16"
D_BITS="1 2 4 8 16"
SSM_ACT_BITS="1 2 3 4 5 6 7 8 16 None"
TAUS=`ls -1 dynamical/data/MackeyGlass/`

if [ $# -ne 1 ]; then
   echo "Usage: $0 <case>"
   exit 1
fi

CASE=$1

for tau in $TAUS; do
   if [ $CASE -eq 0 ]; then
      for a_bits in $A_BITS; do  # 1: A matrix
         echo "Running Mackey-Glass system with a_bits=$a_bits, tau=$tau"
         python3 run_experiment.py --a_bits $a_bits --experiment mackey_glass --tau $tau
      done
   elif [ $CASE -eq 1 ]; then # 2: Activation
      for ssm_act_bits in $SSM_ACT_BITS; do
         echo "Running Mackey-Glass system with ssm_act_bits=$ssm_act_bits, tau=$tau"
         python3 run_experiment.py --ssm_act_bits $ssm_act_bits --experiment mackey_glass --tau $tau
      done
   elif [ $CASE -eq 2 ]; then # 3: Everything, except A
      for bits in $A_BITS; do
         echo "Running Mackey-Glass system with ALL bits=$bits, except A=fp32"
         python3 run_experiment.py --b_bits $bits --c_bits $bits --d_bits $bits --ssm_act_bits $bits \
               --experiment mackey_glass --tau $tau
      done
   elif [ $CASE -eq 3 ]; then # 4: Everything, also A
      for bits in $A_BITS; do
         echo "Running Mackey-Glass system with ALL bits=$bits"
         python3 run_experiment.py --a_bits $bits --b_bits $bits --c_bits $bits --d_bits $bits --ssm_act_bits $bits \
               --experiment mackey_glass --tau $tau
      done
   fi
done

# TAUS="tau_1 tau_128 tau_255"
# for tau in $TAUS; do
#    for a_bits in $A_BITS; do
#       for ssm_act_bits in $SSM_ACT_BITS; do
#         echo "Running Mackey-Glass system with a_bits=$a_bits, ssm_act_bits=$ssm_act_bits, tau=$tau"
#         python3 run_experiment.py --a_bits $a_bits --ssm_act_bits $ssm_act_bits --experiment mackey_glass --tau $tau
#       done
#    done
# done
