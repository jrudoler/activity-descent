#!/usr/bin/env bash
# Source from a Slurm job after changing to SLURM_SUBMIT_DIR.
# Run committed science code from an immutable per-job archive so work in the
# checkout can continue while queued or running experiments retain their source.

inference_project_dir=$PWD
inference_snapshot_dir="$inference_project_dir/data/generated/inference/source-${SLURM_JOB_ID:?}"
if [[ ! -f "$inference_snapshot_dir/SOURCE_REVISION" ]]; then
    inference_staging_dir=$(mktemp -d "$inference_project_dir/data/generated/inference/.source-${SLURM_JOB_ID}.XXXXXX")
    inference_revision=$(git rev-parse HEAD)
    git archive "$inference_revision" analysis/learning_rule_inference pyproject.toml uv.lock \
        | tar -x -C "$inference_staging_dir"
    printf '%s\n' "$inference_revision" > "$inference_staging_dir/SOURCE_REVISION"
    mv "$inference_staging_dir" "$inference_snapshot_dir"
fi
export PYTHONPATH="$inference_snapshot_dir"
inference_python() {
    "$inference_project_dir/.venv/bin/python" -P "$@"
}
