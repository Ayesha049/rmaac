#!/usr/bin/env bash
# run_score_experiments.sh
#
# Phase 1: Score-based Langevin correction — train and evaluate for all 6
# MPE scenarios.
#
# Steps per scenario:
#   1. Collect joint trajectory data (if not already present).
#   2. Train score network with DSM (train_score mode).
#   3. Evaluate score correction across the act × obs noise grid (test_score).
#      Sweeps n_steps ∈ {1, 3} at fixed sigma_est and eta.
#   4. Move per-scenario CSV to results/score_sweep/.
#
# Overridable env vars:
#   MODEL_DIR            base directory for saved TF policy checkpoints (default: ./model)
#   SCORE_MODEL_DIR      directory for score network .pt files          (default: ./model/score)
#   DATA_DIR             directory for joint trajectory .npz files      (default: .)
#   RESULT_DIR           output directory for CSVs                      (default: ./results/score_sweep)
#   LOG_DIR              log file directory                              (default: ./logs/score_sweep)
#   NUM_COLLECT_EP       episodes to collect for DSM training data      (default: 5000)
#   DIFFUSION_EPOCHS     DSM training epochs                            (default: 200)
#   NUM_TEST_EPISODES    episodes per evaluation cell                   (default: 800)
#   SCORE_SIGMA_EST      sigma_est used at inference                    (default: 0.5)
#   SCORE_ETA            Langevin step size eta                         (default: 0.1)
#   ANCHOR_TYPE          anchor type for score network                  (default: landmarks+roles)
#   ACT_NOISE_LIST       space-separated act noise std values           (default: 0.0 0.4 0.8 1.2 1.6 2.0)
#   OBS_NOISE_LIST       space-separated obs noise std values           (default: 0.0 0.4 0.8 1.2)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

MODEL_DIR="${MODEL_DIR:-./model}"
SCORE_MODEL_DIR="${SCORE_MODEL_DIR:-./model/score}"
DATA_DIR="${DATA_DIR:-.}"
RESULT_DIR="${RESULT_DIR:-./results/score_sweep}"
LOG_DIR="${LOG_DIR:-./logs/score_sweep}"
NUM_COLLECT_EP="${NUM_COLLECT_EP:-5000}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-200}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"
SCORE_SIGMA_EST="${SCORE_SIGMA_EST:-0.5}"
SCORE_ETA="${SCORE_ETA:-0.1}"
ANCHOR_TYPE="${ANCHOR_TYPE:-landmarks+roles}"
ACT_NOISE_LIST="${ACT_NOISE_LIST:-0.0 0.4 0.8 1.2 1.6 2.0}"
OBS_NOISE_LIST="${OBS_NOISE_LIST:-0.0 0.4 0.8 1.2}"

mkdir -p "$SCORE_MODEL_DIR" "$RESULT_DIR" "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

SCENARIOS=(
    # "simple_tag|0"
    "simple_push|0"
    # "simple_adversary|0"
    # "simple_spread|0"
    # "simple_speaker_listener|0"
    # "simple_crypto|0"
)

for entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r scenario num_adv <<< "$entry"

    exp="${scenario}_exp"
    exp_best="${exp}best"
    model_dir_s="${MODEL_DIR}/${exp}"
    data_path="${DATA_DIR}/joint_diffusion_data_${scenario}.npz"
    score_model_path="${SCORE_MODEL_DIR}/${scenario}_score_${ANCHOR_TYPE}.pt"

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] Scenario: ${scenario}  (anchor: ${ANCHOR_TYPE})"

    # ── Step 1: Collect joint trajectory data ────────────────────
    if [[ ! -f "$data_path" ]]; then
        echo "[$(timestamp)] Collecting ${NUM_COLLECT_EP} episodes of joint data..."
        python -u train.py \
            --scenario             "$scenario" \
            --mode                 collect_joint_diffusion \
            --exp-name             "$exp" \
            --save-dir             "$model_dir_s" \
            --load-dir             "$model_dir_s" \
            --num-adversaries      "$num_adv" \
            --num-collect-episodes "$NUM_COLLECT_EP" \
            --joint-diffusion-data-path "$data_path" \
            --anchor-type          "$ANCHOR_TYPE" \
            2>&1 | tee "$LOG_DIR/${scenario}.collect.log"
        echo "[$(timestamp)] Data saved to ${data_path}"
    else
        echo "[$(timestamp)] Joint data found — skipping collection."
    fi

    # ── Step 2: Train score network ───────────────────────────────
    if [[ ! -f "$score_model_path" ]]; then
        echo "[$(timestamp)] Training score network (DSM, ${DIFFUSION_EPOCHS} epochs)..."
        python -u train.py \
            --scenario                  "$scenario" \
            --mode                      train_score \
            --exp-name                  "$exp" \
            --save-dir                  "$model_dir_s" \
            --num-adversaries           "$num_adv" \
            --joint-diffusion-data-path "$data_path" \
            --anchor-type               "$ANCHOR_TYPE" \
            --score-model-path          "$score_model_path" \
            --score-sigma-min           0.01 \
            --score-sigma-max           3.0 \
            --diffusion-epochs          "$DIFFUSION_EPOCHS" \
            --diffusion-lr              1e-4 \
            --diffusion-batch-size      64 \
            2>&1 | tee "$LOG_DIR/${scenario}.train_score.log"
        echo "[$(timestamp)] Score model saved to ${score_model_path}"
    else
        echo "[$(timestamp)] Score model found — skipping training."
    fi

    # ── Step 3: Evaluate score correction (noise sweep) ──────────
    echo "[$(timestamp)] Running score sweep (act × obs grid)..."
    python -u train.py \
        --scenario                  "$scenario" \
        --mode                      test_score \
        --exp-name                  "$exp" \
        --save-dir                  "$model_dir_s" \
        --load-dir                  "$model_dir_s" \
        --num-adversaries           "$num_adv" \
        --num-test-episodes         "$NUM_TEST_EPISODES" \
        --act-noise-list            $ACT_NOISE_LIST \
        --obs-noise-list            $OBS_NOISE_LIST \
        --joint-diffusion-data-path "$data_path" \
        --anchor-type               "$ANCHOR_TYPE" \
        --score-model-path          "$score_model_path" \
        --score-sigma-est           "$SCORE_SIGMA_EST" \
        --score-eta                 "$SCORE_ETA" \
        --lam-q                     0.0 \
        2>&1 | tee "$LOG_DIR/${scenario}.eval_score.log"

    # ── Step 4: Move CSV to results directory ─────────────────────
    csv_src="${exp_best}_score_sweep_${ANCHOR_TYPE}.csv"
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
