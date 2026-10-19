"""Compare CUDA probes against the exact saved CPU checkpoints and observations.

Training and perturbation random-number streams differ between devices. This
check therefore restores saved CPU states and compares deterministic update
banks; it does not mistake a CUDA retraining run for a paired numeric check.
"""

import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import campaign, core
from .evaluate import write_csv
from .mean_response import slice_state
from .provenance import digest, provenance, save_json


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    metadata = provenance("cuda")
    started = time.perf_counter()
    source = Path("data/generated/inference/confirmatory-v1")
    with Path("results/data/inference_evaluation.csv").open() as stream:
        cutoffs = {int(r["checkpoint_step"]):float(r["cutoff"]) for r in csv.DictReader(stream)
                   if r["profile"]=="primary" and r["method"]=="operator" and float(r["response_noise"])==0}
    measurements, manifests = {}, []
    for shard in range(10):
        path = source / f"primary-evaluation-s{shard:02}.json"
        parent = json.loads(path.read_text())
        if digest(parent["arrays"]) != parent["arrays_sha256"]:
            raise ValueError("Corrupt CPU checkpoint arrays")
        manifests.append({"path":str(path), "sha256":digest(path), "arrays_sha256":parent["arrays_sha256"]})
        data = torch.load(parent["arrays"], weights_only=True, map_location="cpu")
        n = len(parent["seeds"])
        initial = slice_state(data["initial"], 0, n, "cuda")
        errors, norms = data["probe_errors"].cuda(), data["magnitude_errors"].cuda()
        x = data["inputs"].cuda().expand(-1, errors.shape[1], -1)
        for checkpoint in data["checkpoints"]:
            step = checkpoint["step"]
            state = slice_state(checkpoint["state"], 0, n, "cuda")
            for label, rule, mixture in (("gd", "gd", None), ("preconditioned", "preconditioned", None),
                                          ("feedback-0.1", "feedback", .1), ("feedback-1", "feedback", 1.)):
                if mixture is None:
                    current = state
                else:
                    feedback = campaign.align_feedback(initial, mixture)
                    current = replace(state, feedback=feedback.feedback, readout_feedback=feedback.readout_feedback)
                g, v, _ = campaign.matched_observation(current, x, errors, norms, rule, parent["configuration"]["eta"])
                cpu = next(r for r in data["observations"] if r["checkpoint_step"]==step and r["rule"]==label and r["response_noise"]==0)
                torch.testing.assert_close(g.cpu(), cpu["g"], rtol=1e-10, atol=1e-12)
                relative = torch.linalg.vector_norm((v.cpu()-cpu["v"]).flatten(1),dim=-1) / torch.linalg.vector_norm(cpu["v"].flatten(1),dim=-1).clamp_min(1e-30)
                if relative.max() > 1e-7:
                    raise AssertionError(f"CUDA response differs from CPU beyond float64 tolerance: {relative.max()}")
                gpu_scores = campaign.score_observation(g, v, 2*parent["profile"]["outputs"])["scores"]["operator"].cpu().numpy()
                reference = next(r for r in parent["results"] if r["checkpoint_step"]==step and r["rule"]==label and r["response_noise"]==0)
                cpu_scores = np.asarray(reference["scores"]["operator"])
                group = measurements.setdefault((step,label), {"relative":[], "score_error":[], "disagree":[]})
                group["relative"].extend(relative.tolist())
                group["score_error"].extend(np.abs(gpu_scores-cpu_scores).tolist())
                group["disagree"].extend(((gpu_scores>cutoffs[step]) != (cpu_scores>cutoffs[step])).tolist())
        print(f"Verified CUDA against saved CPU shard {shard}",flush=True)
    rows=[]
    for (step,rule), group in measurements.items():
        if len(group["relative"])!=200:
            raise ValueError("Incomplete CUDA/CPU comparison")
        rows.append({"checkpoint_step":step, "rule":rule, "networks":200,
                     "median_relative_response_error":np.median(group["relative"]),
                     "max_relative_response_error":max(group["relative"]),
                     "max_operator_score_error":max(group["score_error"]),
                     "decision_disagreements":sum(group["disagree"])})
    output=Path("results/data/inference_backend_validation.csv")
    write_csv(output,rows)
    save_json(output.with_suffix(".provenance.json"), {"provenance":metadata,
        "seconds":time.perf_counter()-started, "parents":manifests,
        "comparison":"Same saved CPU checkpoints, target probes, magnitude bank, and four deterministic update rules; float64",
        "peak_allocated_bytes":provenance("cuda")["peak_allocated_bytes"]})


if __name__ == "__main__":
    main()
