"""Evaluate completed CUDA replications with the original locked CPU cutoffs."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .evaluate import METHODS, source_signature, wilson, write_csv
from .provenance import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/generated/inference/b200-primary-evaluation"))
    parser.add_argument("--cpu-evaluation", type=Path, default=Path("results/data/inference_evaluation.csv"))
    parser.add_argument("--configuration", type=Path, default=Path("configs/inference/confirmatory_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/data/inference_gpu_evaluation.csv"))
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    with args.cpu_evaluation.open() as stream:
        cpu = {(int(r["checkpoint_step"]), float(r["response_noise"]), r["rule"], r["method"]):r
               for r in csv.DictReader(stream) if r["profile"]=="primary" and int(r["checkpoint_step"])>=0}
    source_manifest = json.loads(args.cpu_evaluation.with_suffix(".provenance.json").read_text())
    groups, manifests, devices = {}, [], []
    for shard in range(10):
        path = args.input_dir / f"primary-evaluation-s{shard:02}.json"
        d = json.loads(path.read_text())
        if d["profile_index"]!=0 or d["split"]!="evaluation" or d["shard"]!=shard:
            raise ValueError(f"Unexpected CUDA shard identity: {path}")
        if d["configuration_sha256"]!=digest(args.configuration) or source_signature(d)!=source_manifest["science_sources"]:
            raise ValueError(f"CUDA/CPU scientific protocol or source mismatch: {path}")
        if not d["provenance"]["device"].startswith("cuda") or "B200" not in d["provenance"].get("gpu", ""):
            raise ValueError("GPU evidence requires actual B200 CUDA execution")
        if digest(d["arrays"])!=d["arrays_sha256"]:
            raise ValueError(f"CUDA arrays failed verification: {path}")
        first = config["splits"]["evaluation"]["seed_start"]+shard*config["shard_size"]
        if d["seeds"]!=list(range(first,first+config["shard_size"])):
            raise ValueError("Unexpected CUDA evaluation seeds")
        keys=[]
        for r in d["results"]:
            for method in METHODS:
                key=r["checkpoint_step"],r["response_noise"],r["rule"],method
                keys.append(key)
                group=groups.setdefault(key,{})
                for seed,value in zip(d["seeds"],r["scores"][method],strict=True):
                    if seed in group:
                        raise ValueError("Duplicated CUDA network")
                    group[seed]=math.nan if value is None else value
        if len(keys)!=len(cpu) or set(keys)!=set(cpu):
            raise ValueError("Incomplete CUDA scientific conditions")
        manifests.append({"path":str(path),"sha256":digest(path),"arrays_sha256":d["arrays_sha256"],
                          "seconds":d["seconds"],"job_id":d["provenance"]["slurm_job_id"],
                          "source_revision":d["provenance"]["analysis_commit"]})
        devices.append({k:d["provenance"].get(k) for k in ("gpu","gpu_capability","gpu_total_memory","driver","torch","cuda_runtime")})
    rows=[]
    for key, values in sorted(groups.items()):
        if len(values)!=200:
            raise ValueError("Incomplete CUDA evaluation sample")
        reference=cpu[key]
        a=np.asarray(list(values.values()));valid=a[np.isfinite(a)]
        cutoff=float(reference["cutoff"])
        count=int((valid>cutoff).sum());low,high=wilson(count,len(valid))
        rows.append({"checkpoint_step":key[0],"response_noise":key[1],"rule":key[2],"method":key[3],
            "cpu_cutoff":cutoff,"evaluation_runs":len(valid),"undefined_runs":len(a)-len(valid),
            "rejections":count,"rejection_rate":count/len(valid) if len(valid) else math.nan,
            "ci_low":low,"ci_high":high,"cpu_rejections":reference["rejections"],
            "cpu_evaluation_runs":reference["evaluation_runs"],"cpu_rejection_rate":reference["rejection_rate"]})
    write_csv(args.output,rows)
    save_json(args.output.with_suffix(".provenance.json"),{
        "configuration_sha256":digest(args.configuration),"evaluator_sha256":digest(__file__),
        "cpu_evaluation_sha256":digest(args.cpu_evaluation),"completed_shards":manifests,
        "devices":devices,"science_sources":source_manifest["science_sources"],
        "interpretation":"CUDA retraining uses a different RNG stream; shared initialization identities do not imply identical trajectories. CPU cutoffs remain fixed."})
    print(f"Verified ten actual B200 shards; saved {len(rows)} CUDA held-out conditions")


if __name__ == "__main__":
    main()
