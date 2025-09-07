#!/bin/bash

TASKNAME=sim_transfer_cube_scripted # keep the same to the key in configs.constants.TASK_CONFIG
POLICYCONFIG=configs/policy/act.yaml # the policy config file path
TRAININGCONFIG=configs/training/default.yaml # the training config file path
NORM=zscore # the normalizer name, e.g., zscore, minmax, and
OUTPUT=ckpt/act_${TASKNAME}_${NORM}_example # the checkpoint name that will be used for evaluation
# Core training parameters are now in HyperArguments (command-line suitable)
# Stable process parameters are in training config file
# Task-related parameters (action_dim, state_dim, chunk_size, etc.) are loaded from task config
python ./train.py \
    --task_name $TASKNAME \
    --policy_config $POLICYCONFIG \
    --training_config $TRAININGCONFIG \
    --output_dir $OUTPUT \
