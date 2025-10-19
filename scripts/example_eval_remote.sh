#!/bin/bash
# Example script for remote policy evaluation

# Server configuration
SERVER_HOST="localhost"
SERVER_PORT="5000"
SERVER_ADDRESS="${SERVER_HOST}:${SERVER_PORT}"

# Evaluation configuration
ENV="aloha"
NUM_ROLLOUT=4
NUM_ENVS=2
CHUNK_SIZE=64
OUTPUT_DIR="results/remote_eval_$(date +%Y%m%d_%H%M%S)"

echo "🤖 Remote Policy Evaluation Example"
echo "=================================="
echo "Server: $SERVER_ADDRESS"
echo "Environment: $ENV"
echo "Rollouts: $NUM_ROLLOUT"
echo "Parallel Envs: $NUM_ENVS"
echo "Chunk Size: $CHUNK_SIZE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if server is running
echo "🔍 Checking if policy server is running..."
if ! nc -z $SERVER_HOST $SERVER_PORT 2>/dev/null; then
    echo "❌ Policy server is not running at $SERVER_ADDRESS"
    echo ""
    echo "Please start the policy server first:"
    echo "  python start_policy_server.py --host $SERVER_HOST --port $SERVER_PORT"
    echo "  OR"
    echo "  ./scripts/start_policy_server.sh --host $SERVER_HOST --port $SERVER_PORT"
    exit 1
fi

echo "✅ Policy server is running at $SERVER_ADDRESS"
echo ""

# Run remote evaluation
echo "🚀 Starting remote evaluation..."
python eval_remote.py \
    --model_name_or_path $SERVER_ADDRESS \
    --env $ENV \
    --num_rollout $NUM_ROLLOUT \
    --num_envs $NUM_ENVS \
    --chunk_size $CHUNK_SIZE \
    --output_dir $OUTPUT_DIR \
    --fps 50

echo ""
echo "✅ Remote evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
