#!/usr/bin/env bash
# run_joint_noise_sweep_m3ddpg.sh
#
# Same joint act/obs noise sweep as run_joint_noise_sweep.sh but trains and
# evaluates M3DDPG policies (--variant m3ddpg).
#
# For each of the 6 MPE scenarios and each (act_noise_std, obs_noise_std) pair,
# evaluates three conditions: act-only noise, obs-only noise, and joint noise.
# Trains from scratch with M3DDPG if no best checkpoint is found.
#
# Usage:
#   bash run_joint_noise_sweep_m3ddpg.sh
#
# Overridable env vars:
#   MODEL_DIR          base directory for saved models        (default: ./model)
#   NUM_TEST_EPISODES  episodes per evaluation cell           (default: 800)
#   TRAIN_EPISODES     episodes to train if needed            (default: 30000)
#   SAVE_RATE          checkpoint save interval               (default: 500)
#   ACT_NOISE_LIST     space-separated act noise std values   (default: 0.0 0.4 0.8 1.2 1.6 2.0)
#   OBS_NOISE_LIST     space-separated obs noise std values   (default: 0.0 0.4 0.8 1.2)
#   ADV_EPS            M3DDPG perturbation eps for good agents (default: 1e-3)
#   ADV_EPS_S          M3DDPG perturbation eps for adversaries (default: 1e-5)
#
# Outputs (one CSV per scenario in results/noise_sweep_m3ddpg/):
#   <scenario>_m3ddpg_expbest_noise_sweep.csv
#     columns: act_noise_std, obs_noise_std, act_only_reward, obs_only_reward, joint_reward
#   logs/joint_noise_sweep_m3ddpg/<scenario>.{train,eval}.log

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
ADV_EPS="${ADV_EPS:-1e-3}"
ADV_EPS_S="${ADV_EPS_S:-1e-5}"

LOG_DIR="./logs/joint_noise_sweep_m3ddpg"
RESULT_DIR="./results/noise_sweep_m3ddpg"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

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
    exp="${scenario}_m3ddpg_exp"
    exp_best="${exp}best"
    model_dir_s="${MODEL_DIR}/${exp}"

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] Scenario: ${scenario}  (variant: m3ddpg)"

    # ── Step 1: Train if no best checkpoint exists ────────────────
    if [[ ! -f "${model_dir_s}/${exp_best}.index" ]]; then
        echo "[$(timestamp)] No checkpoint found — training ${scenario} (m3ddpg) for ${TRAIN_EPISODES} episodes..."
        mkdir -p "$model_dir_s"
        python -u train.py \
            --scenario        "$scenario" \
            --mode            train \
            --variant         m3ddpg \
            --adv-eps         "$ADV_EPS" \
            --adv-eps-s       "$ADV_EPS_S" \
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
        --variant           m3ddpg \
        --adv-eps           "$ADV_EPS" \
        --adv-eps-s         "$ADV_EPS_S" \
        --exp-name          "$exp" \
        --save-dir          "$model_dir_s" \
        --load-dir          "$model_dir_s" \
        --num-adversaries   "$num_adv" \
        --num-test-episodes "$NUM_TEST_EPISODES" \
        --act-noise-list    $ACT_NOISE_LIST \
        --obs-noise-list    $OBS_NOISE_LIST \
        2>&1 | tee "$LOG_DIR/${scenario}.eval.log"

    # ── Step 3: Move CSV to results directory ─────────────────────
    csv_src="${exp_best}_noise_sweep.csv"
    if [[ -f "$csv_src" ]]; then
        mv "$csv_src" "${RESULT_DIR}/${csv_src}"
        echo "[$(timestamp)] Results saved to ${RESULT_DIR}/${csv_src}"
    else
        echo "[$(timestamp)] WARNING: expected CSV ${csv_src} not found."
    fi
done

echo ""
echo "============================================================"
echo "All done. Per-scenario CSVs in: ${RESULT_DIR}/"
echo "Logs: ${LOG_DIR}/"
