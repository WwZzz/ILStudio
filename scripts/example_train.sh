#!/bin/bash

TASKNAME=sim_transfer_cube_scripted # keep the same to the key in configs.constants.TASK_CONFIG
POLICYCONFIG=configs/policy/act.yaml # the policy config file path
TRAININGCONFIG=configs/training/default.yaml # the training config file path
OUTPUT=ckpt/act_${TASKNAME}_example # the checkpoint name that will be used for evaluation

python ./train.py \
    --task_name $TASKNAME \
    --policy_config $POLICYCONFIG \
    --training_config $TRAININGCONFIG \
    --output_dir $OUTPUT \
