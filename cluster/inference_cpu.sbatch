#!/usr/bin/env bash
#SBATCH --job-name=activity-inference-cpu
#SBATCH --account=wharton-phd-students
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=wharton-phd-student-genoa
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
#SBATCH --output=data/generated/inference/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the analysis repository root}"
source cluster/inference_env.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mode=${1:-preflight}
if (( $# )); then shift; fi
case "$mode" in
    preflight)
        inference_python -m analysis.learning_rule_inference.validate --device cpu \
            --output "data/generated/inference/validation-cpu-${SLURM_JOB_ID}.json"
        for networks in 1 20 64; do
            inference_python -m analysis.learning_rule_inference.run benchmark \
                --device cpu --networks "$networks" --width 64 \
                --output "data/generated/inference/benchmark-cpu-n${networks}-${SLURM_JOB_ID}.json" "$@"
        done
        ;;
    validate)
        inference_python -m analysis.learning_rule_inference.validate --device cpu "$@"
        ;;
    benchmark|pilot)
        inference_python -m analysis.learning_rule_inference.run "$mode" --device cpu "$@"
        ;;
    campaign)
        inference_python -m analysis.learning_rule_inference.campaign --device cpu "$@"
        ;;
    *) printf 'Unknown experiment mode: %s\n' "$mode" >&2; exit 2 ;;
esac
