#!/bin/bash
# run_joint_denoiser_experiments.sh
#
# Complete pipeline for the joint state-action diffusion denoiser.
# Can be run from any directory.
#
# Usage:
#   bash run_joint_denoiser_experiments.sh <scenario> [exp_name]
#   Example: bash run_joint_denoiser_experiments.sh simple_tag simple_tag_run1

set -e

# Always run from the experiments/ directory regardless of where the script is invoked from
cd "$(dirname "$0")"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO=${1:-simple_push}
EXP=${2:-${SCENARIO}_exp}

MODEL_DIR="./model/${EXP}"
DATA_PATH="./joint_diffusion_data_${EXP}.npz"
RESULT_DIR="./results/${EXP}"

mkdir -p "${MODEL_DIR}" "${RESULT_DIR}"

# ============================================================
# Step 1: Train MADDPG policy (skip if checkpoint exists)
# ============================================================
BEST_CHECKPOINT_PREFIX="${MODEL_DIR}/${EXP}best"
if [ ! -f "${BEST_CHECKPOINT_PREFIX}.index" ]; then
    echo "=== [Step 1] Training MADDPG on ${SCENARIO} ==="
    python train.py \
        --scenario "${SCENARIO}" \
        --mode train \
        --exp-name "${EXP}" \
        --save-dir "${MODEL_DIR}" \
        --num-episodes 30000
else
    echo "=== [Step 1] Skipping training (checkpoint exists at ${MODEL_DIR}) ==="
fi

# ============================================================
# Step 2: Collect joint (state + action + anchor) trajectories
# ============================================================
if [ ! -f "${DATA_PATH}" ]; then
    echo "=== [Step 2] Collecting joint diffusion data ==="
    python train.py \
        --scenario "${SCENARIO}" \
        --mode collect_joint_diffusion \
        --exp-name "${EXP}" \
        --save-dir "${MODEL_DIR}" \
        --num-collect-episodes 5000 \
        --joint-diffusion-data-path "${DATA_PATH}" \
        --diffusion-horizon 25
else
    echo "=== [Step 2] Skipping data collection (exists at ${DATA_PATH}) ==="
fi

# ============================================================
# Step 3: Train joint denoisers (one per anchor type)
# ============================================================
for ANCHOR in none init_obs landmarks landmarks+roles; do
    MODEL_PATH="${RESULT_DIR}/denoiser_${ANCHOR}.pt"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "=== [Step 3] Training joint denoiser (anchor=${ANCHOR}) ==="
        python train.py \
            --scenario "${SCENARIO}" \
            --mode train_joint_denoiser \
            --exp-name "${EXP}" \
            --save-dir "${MODEL_DIR}" \
            --joint-diffusion-data-path "${DATA_PATH}" \
            --joint-denoiser-model-path "${MODEL_PATH}" \
            --anchor-type "${ANCHOR}" \
            --denoiser-hidden-dim 256 \
            --denoiser-n-blocks 4 \
            --diffusion-steps 100 \
            --diffusion-epochs 200 \
            --diffusion-lr 1e-4 \
            --diffusion-batch-size 128 \
            --channel-weight-lambda 0.1
    else
        echo "=== [Step 3] Skipping denoiser training (anchor=${ANCHOR}, exists) ==="
    fi
done

# ============================================================
# Step 4: Evaluate with test_joint sweep
# ============================================================
for ANCHOR in none init_obs landmarks landmarks+roles; do
    MODEL_PATH="${RESULT_DIR}/denoiser_${ANCHOR}.pt"

    echo "=== [Step 4] Evaluating joint denoiser (anchor=${ANCHOR}) ==="
    python train.py \
        --scenario "${SCENARIO}" \
        --mode test_joint \
        --exp-name "${EXP}" \
        --save-dir "${MODEL_DIR}" \
        --joint-denoiser-model-path "${MODEL_PATH}" \
        --anchor-type "${ANCHOR}" \
        --num-test-episodes 400
done

echo "=== Done! Results saved as CSV files in current directory ==="
