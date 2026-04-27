#!/usr/bin/env bash
# run_joint_noise_sweep.sh
#
# For each of the 6 MPE scenarios and each (act_noise_std, obs_noise_std) pair,
# evaluates three conditions: act-only noise, obs-only noise, and joint noise.
# Trains from scratch if no best checkpoint is found.
#
# Usage:
#   bash run_joint_noise_sweep.sh
#
# Overridable env vars:
#   MODEL_DIR          base directory for saved models        (default: ./model)
#   NUM_TEST_EPISODES  episodes per evaluation cell           (default: 200)
#   TRAIN_EPISODES     episodes to train if needed            (default: 30000)
#   SAVE_RATE          checkpoint save interval               (default: 1000)
#   ACT_NOISE_LIST     space-separated act noise std values   (default: 0.0 0.4 0.8 1.2 1.6 2.0)
#   OBS_NOISE_LIST     space-separated obs noise std values   (default: 0.0 0.4 0.8 1.2)
#
# Outputs (one CSV per scenario):
#   <scenario>__cleanbest_noise_sweep.csv
#     columns: act_noise_std, obs_noise_std, act_only_reward, obs_only_reward, joint_reward
#   logs/joint_noise_sweep/<scenario>.{train,eval}.log

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

MODEL_DIR="${MODEL_DIR:-./model}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"
TRAIN_EPISODES="${TRAIN_EPISODES:-30000}"
SAVE_RATE="${SAVE_RATE:-500}"
ACT_NOISE_LIST="${ACT_NOISE_LIST:-0.0 0.4 0.8 1.2 1.6 2.0}"
OBS_NOISE_LIST="${OBS_NOISE_LIST:-0.0 0.4 0.8 1.2}"

LOG_DIR="./logs/joint_noise_sweep"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# 6 MPE scenarios: "scenario_name|num_adversaries"
# num_adversaries matches run_joint_denoiser_experiments.sh (no --num-adversaries
# flag passed there, so the argparse default of 0 applies to all scenarios).
SCENARIOS=(
    "simple_tag|0"
    "simple_push|0"
    "simple_adversary|0"
    "simple_spread|0"
    "simple_speaker_listener|0"
    "simple_crypto|0"
)

for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r scenario num_adv <<< "$entry"
    # Mirror run_joint_denoiser_experiments.sh: EXP=${2:-${SCENARIO}_exp}
    exp="${scenario}_exp"
    exp_best="${exp}best"
    model_dir_s="${MODEL_DIR}/${exp}"

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] Scenario: ${scenario}"

    # ── Step 1: Train if no best checkpoint exists ────────────────
    if [[ ! -f "${model_dir_s}/${exp_best}.index" ]]; then
        echo "[$(timestamp)] No checkpoint found — training ${scenario} for ${TRAIN_EPISODES} episodes..."
        mkdir -p "$model_dir_s"
        python -u train.py \
            --scenario        "$scenario" \
            --mode            train \
            --adv-type        none \
            --exp-name        "$exp" \
            --save-dir        "$model_dir_s" \
            --num-adversaries "$num_adv" \
            --num-episodes    "$TRAIN_EPISODES" \
            --save-rate       "$SAVE_RATE" \
            2>&1 | tee "$LOG_DIR/${scenario}.train.log"
        echo "[$(timestamp)] Training done."
    else
        echo "[$(timestamp)] Checkpoint found — skipping training."
    fi

    # ── Step 2: Noise sweep evaluation ───────────────────────────
    echo "[$(timestamp)] Running noise sweep (act × obs grid)..."
    python -u train.py \
        --scenario          "$scenario" \
        --mode              noise_sweep \
        --exp-name          "$exp" \
        --save-dir          "$model_dir_s" \
        --load-dir          "$model_dir_s" \
        --num-adversaries   "$num_adv" \
        --num-test-episodes "$NUM_TEST_EPISODES" \
        --act-noise-list    $ACT_NOISE_LIST \
        --obs-noise-list    $OBS_NOISE_LIST \
        2>&1 | tee "$LOG_DIR/${scenario}.eval.log"

    echo "[$(timestamp)] Results saved to ${exp_best}_noise_sweep.csv"
done

echo ""
echo "============================================================"
echo "All done. Per-scenario CSVs: <scenario>__cleanbest_noise_sweep.csv"
echo "Logs: ${LOG_DIR}/"
