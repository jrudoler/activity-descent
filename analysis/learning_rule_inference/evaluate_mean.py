"""Evaluate the locked conditional-mean protocol with independent null calibration."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .evaluate import conformal_cutoff, wilson, write_csv
from .mean_response import RULES
from .provenance import digest, save_json

METRICS = ("relative_error_to_deterministic_mean_step", "mc_variability_scale", "mean_order_gap")
SOURCES = ("core.py", "operators.py", "campaign.py", "mean_response.py")


def collect(directory, configuration, configuration_path, parent_configuration):
    groups, controls, manifests, signatures, identities = {}, {}, [], [], set()
    conditions = {(rule, mode, count) for rule in RULES
                  for mode in ("responses-first", "directions-first")
                  for count in configuration["sample_counts"]}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text())
        if report["configuration_sha256"] != digest(configuration_path):
            raise ValueError(f"Conditional-mean protocol mismatch: {path}")
        split, shard = report["split"], report["shard"]
        identity = split, shard
        if identity in identities:
            raise ValueError(f"Duplicate shard: {identity}")
        identities.add(identity)
        n = parent_configuration["shard_size"]
        first = parent_configuration["splits"][split]["seed_start"] + n * shard
        if report["seeds"] != list(range(first, first + n)):
            raise ValueError(f"Unexpected network identities: {path}")
        if digest(report["arrays"]) != report["arrays_sha256"]:
            raise ValueError(f"Corrupted raw arrays: {path}")
        if digest(report["parent"]) != report["parent_sha256"]:
            raise ValueError(f"Changed parent checkpoint report: {path}")
        parent = json.loads(Path(report["parent"]).read_text())
        if parent["seeds"] != report["seeds"] or parent["configuration"] != parent_configuration:
            raise ValueError(f"Parent experiment mismatch: {path}")
        if digest(parent["arrays"]) != parent["arrays_sha256"]:
            raise ValueError(f"Corrupted parent checkpoint arrays: {path}")
        signature = {name: report["provenance"]["sources"][f"analysis/learning_rule_inference/{name}"]
                     for name in SOURCES}
        signatures.append(signature)
        actual = [(r["rule"], r["mode"], r["samples"]) for r in report["results"]]
        if len(actual) != len(conditions) or set(actual) != conditions:
            raise ValueError(f"Missing or duplicated conditions: {path}")
        for row in report["results"]:
            for metric in ("studentized_reciprocity", "raw_operator", *METRICS):
                key = split, row["mode"], row["samples"], row["rule"], metric
                group = groups.setdefault(key, {})
                for seed, value in zip(report["seeds"], row[metric], strict=True):
                    if seed in group:
                        raise ValueError(f"Duplicate seed in {key}")
                    group[seed] = math.nan if value is None else value
        for rule, scores in report["deterministic_null_scores"].items():
            group = controls.setdefault((split, rule), {})
            for seed, value in zip(report["seeds"], scores, strict=True):
                if seed in group:
                    raise ValueError("Duplicated deterministic null")
                group[seed] = math.nan if value is None else value
        manifests.append({"path": str(path), "sha256": digest(path), "arrays_sha256": report["arrays_sha256"],
                          "parent_sha256": report["parent_sha256"], "seconds": report["seconds"],
                          "job_id": report["provenance"]["slurm_job_id"],
                          "source_revision": report["provenance"]["analysis_commit"]})
    expected = {(split, shard) for split in ("calibration", "evaluation")
                for shard in range(configuration["shards_per_split"])}
    if identities != expected:
        raise ValueError(f"Incomplete mean-response experiment: missing {expected - identities}")
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Mean-response scientific source hashes differ")
    for key, values in groups.items():
        if len(values) != parent_configuration["splits"][key[0]]["runs"]:
            raise ValueError(f"Incomplete independent sample: {key}")
    return groups, controls, manifests, signatures[0]


def evaluate(groups, controls, configuration):
    rows, accuracy = [], []
    alpha = configuration["alpha"]
    naive_cutoff = max(conformal_cutoff(list(controls[("calibration", null)].values()), alpha)
                       for null in configuration["null_controls"])
    rng = np.random.default_rng(681292)
    for mode in ("responses-first", "directions-first"):
        for count in configuration["sample_counts"]:
            aware_cutoff = max(conformal_cutoff(list(groups[("calibration", mode, count, null,
                                  "studentized_reciprocity")].values()), alpha)
                               for null in configuration["null_controls"])
            for rule in RULES:
                for method, metric, cutoff in (("uncertainty_aware", "studentized_reciprocity", aware_cutoff),
                                                ("naive_deterministic_cutoff", "raw_operator", naive_cutoff)):
                    values = np.asarray(list(groups[("evaluation", mode, count, rule, metric)].values()))
                    valid = values[np.isfinite(values)]
                    rejected = int((valid > cutoff).sum())
                    lower, upper = wilson(rejected, len(valid))
                    rows.append({"mode": mode, "samples": count, "rule": rule, "method": method,
                        "cutoff": cutoff, "evaluation_runs": len(valid), "undefined_runs": len(values) - len(valid),
                        "rejections": rejected, "rejection_rate": rejected / len(valid) if len(valid) else math.nan,
                        "ci_low": lower, "ci_high": upper})
                for metric in METRICS:
                    values = np.asarray(list(groups[("evaluation", mode, count, rule, metric)].values()))
                    valid = values[np.isfinite(values)]
                    if len(valid):
                        boot = np.median(valid[rng.integers(len(valid), size=(2000, len(valid)))], axis=-1)
                        median, low, high = float(np.median(valid)), *np.quantile(boot, [.025, .975]).tolist()
                    else:
                        median = low = high = math.nan
                    accuracy.append({"mode": mode, "samples": count, "rule": rule, "metric": metric,
                        "evaluation_runs": len(valid), "median": median, "ci_low": low, "ci_high": high})
    return rows, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, default=Path("configs/inference/mean_v1.json"))
    parser.add_argument("--parent-configuration", type=Path, default=Path("configs/inference/confirmatory_v1.json"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/data/inference_mean_evaluation.csv"))
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    if not config["locked"]:
        raise ValueError("Conditional-mean evaluation requires a locked protocol")
    parent = json.loads(args.parent_configuration.read_text())
    groups, controls, manifests, sources = collect(args.input_dir, config, args.configuration, parent)
    rows, accuracy = evaluate(groups, controls, config)
    write_csv(args.output, rows)
    write_csv(args.output.with_name("inference_mean_accuracy.csv"), accuracy)
    save_json(args.output.with_suffix(".provenance.json"), {
        "configuration": config, "configuration_sha256": digest(args.configuration),
        "science_sources": sources, "completed_shards": manifests,
        "evaluator_sha256": digest(__file__), "intervals": "Wilson 95% rates; 2000 network bootstrap medians",
        "interpretation": "Necessary mean reciprocity test calibrated against two specified stochastic null families"})
    print(f"Verified {len(manifests)} shards and saved {len(rows)} mean-response comparisons")


if __name__ == "__main__":
    main()
