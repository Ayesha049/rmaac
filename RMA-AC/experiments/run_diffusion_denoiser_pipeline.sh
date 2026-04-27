#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/experiments"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/experiments/model}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/experiments/diffusion_runs}"
COLLECT_EPISODES="${COLLECT_EPISODES:-2500}"
DIFFUSION_HORIZON="${DIFFUSION_HORIZON:-25}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-500}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-64}"
DIFFUSION_LR="${DIFFUSION_LR:-1e-4}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-200}"
T_START_LIST=(${T_START_LIST:-20 40 60})
ADVERSARIAL_ADV_TYPE="${ADVERSARIAL_ADV_TYPE:-act}"

# Edit this list to control which scenarios are processed.
SCENARIOS=(${SCENARIOS:-simple_tag})

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

run_cmd() {
    local phase="$1"
    shift

    echo "[$(timestamp)] [${phase}] START"
    echo "[$(timestamp)] [${phase}] COMMAND: $*"
    "$@"
    echo "[$(timestamp)] [${phase}] DONE"
}

merge_npz() {
    local clean_npz="$1"
    local adv_npz="$2"
    local merged_npz="$3"

    python - "$clean_npz" "$adv_npz" "$merged_npz" <<'PY'
import sys
import numpy as np

clean_path, adv_path, merged_path = sys.argv[1:4]
clean = np.load(clean_path)
adv = np.load(adv_path)

states = np.concatenate([clean["states"], adv["states"]], axis=0)
actions = np.concatenate([clean["actions"], adv["actions"]], axis=0)

np.savez(merged_path, states=states, actions=actions)
print("[Diffusion] merged dataset -> {} (states {}, actions {})".format(
    merged_path, states.shape, actions.shape
))
PY
}

get_scenario_config() {
    local scenario="$1"
    case "$scenario" in
        simple_tag)
            printf '%s\n' "--num-adversaries 3"
            ;;
        simple_speaker_listener|simple_adversary|simple_spread|simple_push|simple_crypto)
            printf '%s\n' "--num-adversaries 0"
            ;;
        *)
            echo "Unknown scenario: $scenario" >&2
            return 1
            ;;
    esac
}

mkdir -p "$OUT_DIR"

if [[ "$RUN_TRAINING" != "0" && "$RUN_TRAINING" != "1" ]]; then
    echo "RUN_TRAINING must be 0 or 1" >&2
    exit 1
fi

if [[ "$RUN_EVALUATION" != "0" && "$RUN_EVALUATION" != "1" ]]; then
    echo "RUN_EVALUATION must be 0 or 1" >&2
    exit 1
fi

if [[ "$RUN_TRAINING" == "0" && "$RUN_EVALUATION" == "0" ]]; then
    echo "Nothing to run: set RUN_TRAINING=1 and/or RUN_EVALUATION=1" >&2
    exit 1
fi

for scenario in "${SCENARIOS[@]}"; do
    IFS='|' read -r extra_args <<< "$(get_scenario_config "$scenario")"

    clean_exp="${scenario}__cleanbest"
    adv_exp="${scenario}__adv_actbest"

    clean_data="$OUT_DIR/${scenario}__clean_diffusion.npz"
    adv_data="$OUT_DIR/${scenario}__adv_diffusion.npz"
    merged_data="$OUT_DIR/${scenario}__merged_diffusion.npz"
    model_path="$OUT_DIR/${scenario}__diffusion_model.pt"

    echo "============================================================"
    echo "Scenario: ${scenario}"
    echo "Clean checkpoint: ${clean_exp}"
    echo "Adv checkpoint: ${adv_exp}"

    if [[ "$RUN_TRAINING" == "1" ]]; then
        run_cmd "${scenario}:collect-clean" \
            env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 python -u train.py \
            --scenario "$scenario" \
            --mode collect_diffusion \
            $extra_args \
            --num-episodes "$COLLECT_EPISODES" \
            --diffusion-horizon "$DIFFUSION_HORIZON" \
            --diffusion-data-path "$clean_data" \
            --save-dir "$MODEL_DIR" \
            --exp-name "$clean_exp"

        run_cmd "${scenario}:collect-adv" \
            env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 python -u train.py \
            --scenario "$scenario" \
            --mode collect_diffusion \
            $extra_args \
            --num-episodes "$COLLECT_EPISODES" \
            --diffusion-horizon "$DIFFUSION_HORIZON" \
            --diffusion-data-path "$adv_data" \
            --save-dir "$MODEL_DIR" \
            --exp-name "$adv_exp"

        merge_npz "$clean_data" "$adv_data" "$merged_data"

        run_cmd "${scenario}:train-diffusion" \
            env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 python -u train.py \
            --scenario "$scenario" \
            --mode train_diffusion \
            --diffusion-data-path "$merged_data" \
            --diffusion-model-path "$model_path" \
            --diffusion-horizon "$DIFFUSION_HORIZON" \
            --diffusion-steps "$DIFFUSION_STEPS" \
            --diffusion-epochs "$DIFFUSION_EPOCHS" \
            --diffusion-batch-size "$DIFFUSION_BATCH_SIZE" \
            --diffusion-lr "$DIFFUSION_LR"
    fi

    if [[ "$RUN_EVALUATION" == "1" ]]; then
        if [[ ! -f "$model_path" ]]; then
            echo "Diffusion model not found for ${scenario}: $model_path" >&2
            echo "Run with RUN_TRAINING=1 first, or point OUT_DIR to existing models." >&2
            exit 1
        fi

        run_cmd "${scenario}:evaluate" \
            env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 python -u train.py \
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
            --diffusion-model-path "$model_path" \
            --exp-name "${scenario}__compare_df"
    fi

done

echo "Done. Diffusion data and models are in: $OUT_DIR"