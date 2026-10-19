#!/usr/bin/env bash
#SBATCH --job-name=activity-inference-gpu
#SBATCH --account=wharton-phd-students
#SBATCH --partition=dgx-b200
#SBATCH --qos=wharton-phd-student-dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=data/generated/inference/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the analysis repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mode=${1:-preflight}
if (( $# )); then shift; fi
case "$mode" in
    preflight)
        .venv/bin/python -m analysis.learning_rule_inference.validate --device cuda \
            --output "data/generated/inference/validation-cuda-${SLURM_JOB_ID}.json"
        for networks in 1 20 64; do
            .venv/bin/python -m analysis.learning_rule_inference.run benchmark \
                --device cuda --networks "$networks" --width 64 \
                --output "data/generated/inference/benchmark-cuda-n${networks}-${SLURM_JOB_ID}.json" "$@"
        done
        ;;
    validate)
        exec .venv/bin/python -m analysis.learning_rule_inference.validate --device cuda "$@"
        ;;
    benchmark|pilot)
        exec .venv/bin/python -m analysis.learning_rule_inference.run "$mode" --device cuda "$@"
        ;;
    *) printf 'Unknown experiment mode: %s\n' "$mode" >&2; exit 2 ;;
esac
