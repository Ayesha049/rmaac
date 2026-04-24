#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/experiments"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="$PROJECT_ROOT/experiments/logs/action_noise_ablation"
mkdir -p "$LOG_DIR"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

run_phase() {
    local phase="$1"
    local log_file="$2"
    shift 2

    echo "[$(timestamp)] [${phase}] START"
    set +e
    {
        echo "[$(timestamp)] [${phase}] COMMAND: $*"
        "$@"
    } 2>&1 | tee "$log_file"
    local cmd_rc=${PIPESTATUS[0]}
    set -e

    if [[ $cmd_rc -ne 0 ]]; then
        echo "[$(timestamp)] [${phase}] FAILED (exit code: ${cmd_rc})"
        echo "[$(timestamp)] [${phase}] Last 40 log lines:"
        tail -n 40 "$log_file"
        exit "$cmd_rc"
    fi

    echo "[$(timestamp)] [${phase}] DONE"
}

# Keep evaluation manageable; train.py already sweeps action-noise std internally
# in test mode, so the bash script only needs to call it once per trained model.
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-200}"
SAVE_RATE="${SAVE_RATE:-500}"
BASE_TRAIN_EPISODES="${BASE_TRAIN_EPISODES:-30000}"
TAG_TRAIN_EPISODES="${TAG_TRAIN_EPISODES:-15000}"
ADVERSARIAL_ADV_TYPE="${ADVERSARIAL_ADV_TYPE:-act}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/experiments/model}"
T_START_LIST=(${T_START_LIST:-20 40 60})

# Edit this list to control which scenarios are included in the sweep.
SCENARIOS=(${SCENARIOS:-simple_tag simple_speaker_listener simple_adversary simple_spread simple_push simple_crypto})

get_scenario_config() {
    local scenario="$1"
    case "$scenario" in
        simple_tag)
            printf '%s|%s|%s\n' "$TAG_TRAIN_EPISODES" "$TAG_TRAIN_EPISODES" "--num-adversaries 3"
            ;;
        simple_speaker_listener|simple_adversary|simple_spread|simple_push|simple_crypto)
            printf '%s|%s|%s\n' "$BASE_TRAIN_EPISODES" "$BASE_TRAIN_EPISODES" "--num-adversaries 0"
            ;;
        *)
            echo "Unknown scenario: $scenario" >&2
            return 1
            ;;
    esac
}

for scenario in "${SCENARIOS[@]}"; do
    IFS='|' read -r clean_episodes adv_episodes extra_args <<< "$(get_scenario_config "$scenario")"

    clean_exp="${scenario}__clean"
    adv_exp="${scenario}__adv_act"

    echo "============================================================"
    echo "Scenario: ${scenario}"
    echo "Clean baseline: ${clean_exp}"
    echo "Adversarial training: ${adv_exp}"

    run_phase "${clean_exp}:train" "$LOG_DIR/${clean_exp}.train.log" \
        env CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 python -u train.py \
        --scenario "$scenario" \
        --mode train \
        --adv-type none \
        $extra_args \
        --num-episodes "$clean_episodes" \
        --save-rate "$SAVE_RATE" \
        --save-dir "$MODEL_DIR" \
        --exp-name "$clean_exp"

    run_phase "${adv_exp}:train" "$LOG_DIR/${adv_exp}.train.log" \
        env CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 python -u train.py \
        --scenario "$scenario" \
        --mode train \
        --adv-type "$ADVERSARIAL_ADV_TYPE" \
        $extra_args \
        --num-episodes "$adv_episodes" \
        --save-rate "$SAVE_RATE" \
        --save-dir "$MODEL_DIR" \
        --exp-name "$adv_exp"

    run_phase "${scenario}:compare" "$LOG_DIR/${scenario}.compare.log" \
        env CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 python -u train.py \
        --scenario "$scenario" \
        --mode test \
        --compare-baseline-adv \
        --t-start-list "${T_START_LIST[@]}" \
        --adv-type "$ADVERSARIAL_ADV_TYPE" \
        $extra_args \
        --num-test-episodes "$NUM_TEST_EPISODES" \
        --save-dir "$MODEL_DIR" \
        --baseline-load-dir "$MODEL_DIR" \
        --baseline-exp-name "$clean_exp" \
        --adv-load-dir "$MODEL_DIR" \
        --adv-exp-name "$adv_exp" \
        --exp-name "${scenario}__compare"

done

echo "Done. Logs are in: $LOG_DIR"
