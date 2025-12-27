#!/bin/bash
# Generate datasets for all RLBench tasks
#
# This script automatically generates datasets for all RLBench tasks by:
# 1. Finding all rlbench_*.yaml config files (excluding _ee.yaml versions)
# 2. Extracting task names from config files
# 3. Generating datasets for each task
#
# Usage:
#   bash benchmark/rlbench/generate_all_datasets.sh [--num_demos N] [--base_dir DIR]
#
# Options:
#   --num_demos N    Number of demos to collect per task (default: 50)
#   --base_dir DIR   Base directory for output (default: data/rlbench)
#   --headless       Run in headless mode (default: true)
#   --variation V    Task variation index (default: 0)
#   --skip-existing  Automatically skip tasks with existing datasets (default: true)
#   --force          Force regeneration even if dataset exists
#   --no-confirm     Skip confirmation prompt (default: false)

# Don't exit on error - continue processing other tasks even if one fails
set +e

# Default parameters
NUM_DEMOS=50
BASE_DIR="data/rlbench"
HEADLESS="--headless"
VARIATION=0
SKIP_EXISTING=true
FORCE=false
NO_CONFIRM=false
MAX_ATTEMPTS=300

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_demos)
            NUM_DEMOS="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --no-headless)
            HEADLESS=""
            shift
            ;;
        --variation)
            VARIATION="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --force)
            FORCE=true
            SKIP_EXISTING=false
            shift
            ;;
        --no-confirm)
            NO_CONFIRM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_demos N] [--base_dir DIR] [--headless] [--variation V] [--skip-existing] [--force] [--no-confirm]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Find all RLBench config files (excluding _ee.yaml versions)
CONFIG_DIR="configs/env"
CONFIG_FILES=$(find "$CONFIG_DIR" -name "rlbench_*.yaml" ! -name "*_ee.yaml" | sort)

if [ -z "$CONFIG_FILES" ]; then
    echo "Error: No RLBench config files found in $CONFIG_DIR"
    exit 1
fi

# Count total tasks
TOTAL_TASKS=$(echo "$CONFIG_FILES" | wc -l)
echo "=========================================="
echo "RLBench Dataset Generation Script"
echo "=========================================="
echo "Total tasks to process: $TOTAL_TASKS"
echo "Demos per task: $NUM_DEMOS"
echo "Base output directory: $BASE_DIR"
echo "Headless mode: $([ -n "$HEADLESS" ] && echo "enabled" || echo "disabled")"
echo "Variation index: $VARIATION"
echo "Skip existing datasets: $SKIP_EXISTING"
echo "Force regeneration: $FORCE"
echo "=========================================="
echo ""

# Ask for confirmation unless --no-confirm is set
if [ "$NO_CONFIRM" = false ]; then
    read -p "Continue with dataset generation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Process each config file
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_TASKS=()

for CONFIG_FILE in $CONFIG_FILES; do
    # Extract task name from config file path
    # e.g., configs/env/rlbench_pick_and_lift.yaml -> pick_and_lift
    TASK_NAME=$(basename "$CONFIG_FILE" .yaml | sed 's/^rlbench_//')
    
    # Create output directory name (use task name)
    OUTPUT_DIR="$BASE_DIR/$TASK_NAME"
    
    echo ""
    echo "=========================================="
    echo "Processing: $TASK_NAME"
    echo "Config: $CONFIG_FILE"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="
    
    # Check if dataset already exists
    if [ "$SKIP_EXISTING" = true ] && [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR/*.hdf5 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "⚠️  Dataset already exists at $OUTPUT_DIR - skipping..."
        ((SUCCESS_COUNT++))
        continue
    fi
    
    # Generate dataset
    if python benchmark/rlbench/generate_dataset.py \
        --env_config "$CONFIG_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_demos "$NUM_DEMOS" \
        --variation "$VARIATION" \
        --max_attempts "$MAX_ATTEMPTS" \
        $HEADLESS; then
        echo "✅ Successfully generated dataset for $TASK_NAME"
        ((SUCCESS_COUNT++))
    else
        echo "❌ Failed to generate dataset for $TASK_NAME"
        ((FAIL_COUNT++))
        FAILED_TASKS+=("$TASK_NAME")
        
        # Clean up any leftover CoppeliaSim processes after failure
        pkill -f "coppeliaSim" 2>/dev/null || true
        pkill -f "CoppeliaSim" 2>/dev/null || true
    fi
    
    # Brief pause between tasks to ensure clean state
    sleep 2
done

# Print summary
echo ""
echo "=========================================="
echo "Generation Summary"
echo "=========================================="
echo "Total tasks: $TOTAL_TASKS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failed tasks:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
fi

echo "All datasets saved to: $BASE_DIR"
echo "=========================================="

