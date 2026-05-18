#!/usr/bin/env bash
# Evaluate generated videos on HTEWorld (6 formal metrics: RCBD, LPSA, CISR, PMPA, CPDM, FPHSC).
#
#   OUTPUT_ROOT=/results/my_model \
#   BENCHMARK_ROOT=/data/hteworld/benchmark \
#   bash scripts/eval_hteworld.sh
#
# Parallel sharded evaluation:
#   NUM_SHARDS=8 OUTPUT_ROOT=... BENCHMARK_ROOT=... bash scripts/eval_hteworld.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [[ -z "${OUTPUT_ROOT}" ]]; then
    echo "Error: OUTPUT_ROOT must be set to the directory of generated videos." && exit 1
fi

BENCHMARK_ROOT="${BENCHMARK_ROOT:-benchmark}"
SAVE_DIR="${SAVE_DIR:-results/hteworld_eval}"
MODEL_NAME="${MODEL_NAME:-}"
METRICS="${METRICS:-formal}"
NUM_SHARDS="${NUM_SHARDS:-1}"
CONFIG="${CONFIG:-}"

EXTRA=""
[[ -n "${MODEL_NAME}" ]] && EXTRA="${EXTRA} --model-name ${MODEL_NAME}"
[[ -n "${CONFIG}" ]]     && EXTRA="${EXTRA} --config ${CONFIG}"

echo "HTEWorld eval | Shards=${NUM_SHARDS} | Metrics=${METRICS} | OutputRoot=${OUTPUT_ROOT}"

PIDS=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    python eval/evaluate.py \
        --output-root    "${OUTPUT_ROOT}" \
        --benchmark-root "${BENCHMARK_ROOT}" \
        --save-dir       "${SAVE_DIR}" \
        --metrics        "${METRICS}" \
        --shard-id       "${i}" \
        --num-shards     "${NUM_SHARDS}" \
        ${EXTRA} &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    if [[ -n "${MODEL_NAME}" ]]; then
        SAFE_NAME="$(echo "${MODEL_NAME}" | tr '/\\' '_' | xargs)"
    else
        SAFE_NAME="$(basename "${OUTPUT_ROOT}")"
    fi
    SAFE_NAME="${SAFE_NAME:-model}"

    SHARD_FILES=()
    for ((i = 0; i < NUM_SHARDS; i++)); do
        SHARD_FILES+=("${SAVE_DIR}/${SAFE_NAME}/$(printf '%s_shard_%03d_of_%03d_results.json' "${SAFE_NAME}" "${i}" "${NUM_SHARDS}")")
    done

    MERGED="${SAVE_DIR}/${SAFE_NAME}/${SAFE_NAME}_results.json"
    python eval/merge_sharded_results.py "${SHARD_FILES[@]}" --output "${MERGED}"
    echo "Merged results → ${MERGED}"
fi
