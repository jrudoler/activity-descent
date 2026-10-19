#!/usr/bin/env bash
#SBATCH --job-name=activity-inference-gpu
#SBATCH --account=wharton-phd-students
#SBATCH --partition=dgx-b200
#SBATCH --qos=wharton-phd-student-dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=data/generated/inference/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the analysis repository root}"
source cluster/inference_env.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mode=${1:-preflight}
if (( $# )); then shift; fi
case "$mode" in
    preflight)
        inference_python -m analysis.learning_rule_inference.validate --device cuda \
            --output "data/generated/inference/validation-cuda-${SLURM_JOB_ID}.json"
        for networks in 1 20 64; do
            inference_python -m analysis.learning_rule_inference.run benchmark \
                --device cuda --networks "$networks" --width 64 \
                --output "data/generated/inference/benchmark-cuda-n${networks}-${SLURM_JOB_ID}.json" "$@"
        done
        ;;
    validate)
        inference_python -m analysis.learning_rule_inference.validate --device cuda "$@"
        ;;
    benchmark|pilot)
        inference_python -m analysis.learning_rule_inference.run "$mode" --device cuda "$@"
        ;;
    campaign)
        inference_python -m analysis.learning_rule_inference.campaign --device cuda "$@"
        ;;
    campaign-array)
        inference_python -m analysis.learning_rule_inference.campaign --device cuda \
            --task-index "${SLURM_ARRAY_TASK_ID:?}" "$@"
        ;;
    mean-array)
        inference_python -m analysis.learning_rule_inference.mean_response --device cuda \
            --task-index "${SLURM_ARRAY_TASK_ID:?}" "$@"
        ;;
    replicate-primary)
        inference_python -m analysis.learning_rule_inference.validate --device cuda \
            --output "data/generated/inference/validation-cuda-replication-${SLURM_JOB_ID}.json"
        inference_python -m analysis.learning_rule_inference.backend_check
        for shard in {0..9}; do
            inference_python -m analysis.learning_rule_inference.campaign --device cuda \
                --configuration configs/inference/confirmatory_v1.json \
                --profile 0 --split evaluation --shard "$shard" \
                --output "data/generated/inference/b200-primary-evaluation/primary-evaluation-s$(printf '%02d' "$shard").json"
        done
        ;;
    *) printf 'Unknown experiment mode: %s\n' "$mode" >&2; exit 2 ;;
esac
