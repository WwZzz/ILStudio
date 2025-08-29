#!/bin/bash
# ------------- 默认值 -------------
DEFAULT_SUFFIX="default"
DEFAULT_CKPT="/inspire/hdd/project/robot-action/wangzheng-240308120196/IL-Studio/ckpt/act_insertion_human_zscore"
DEFAULT_POLICY="act"
# ------------- 其余固定配置 -------------
export MUJOCO_GL=egl

ENV=aloha
TASKNAME=insertion
CAM_IDS="[0]"
IMAGE_SIZE="(640,480)"

MODELNAME=act
CHUNKSIZE=100

FPS=50
ROLLOUT=50
PARALLEL=5
MAX_STEPS=400

SUFFIX="${1:-$DEFAULT_SUFFIX}"
CKPT="${2:-$DEFAULT_CKPT}"
MODELNAME="${3:-$DEFAULT_POLICY}"
OUTPUT=results/${MODELNAME}_${ENV}_${TASKNAME}_${SUFFIX}

# ------------- 打印确认 -------------
echo "OUTPUT : $OUTPUT"
echo "CKPT   : $CKPT"

# ------------- 运行评测 -------------
python eval.py \
    --env_name "$ENV" \
    --task "$TASKNAME" \
    --model_name "$MODELNAME" \
    --model_name_or_path "$CKPT" \
    --chunk_size "$CHUNKSIZE" \
    --save_dir "$OUTPUT" \
    --num_rollout "$ROLLOUT" \
    --num_envs "$PARALLEL" \
    --fps "$FPS" \
    --camera_ids "$CAM_IDS" \
    --image_size_primary "$IMAGE_SIZE" \
    --image_size_wrist "$IMAGE_SIZE" \
    --max_timesteps "$MAX_STEPS"