"""Evaluate a locked protocol using complete, independent calibration shards."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .provenance import digest, save_json

METHODS = ("operator", "cosine", "pearson", "downhill", "population", "reciprocity", "linearity")
NULLS = ("gd", "preconditioned")
SCIENCE_SOURCES = ("core.py", "operators.py", "campaign.py", "diagnostics.py")


def conformal_cutoff(values, alpha=.05):
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return math.inf
    order = math.ceil((len(values) + 1) * (1 - alpha))
    return float(values[order - 1]) if order <= len(values) else math.inf


def wilson(successes, count, z=1.959963984540054):
    if count == 0:
        return [math.nan, math.nan]
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def source_signature(report):
    sources = report["provenance"]["sources"]
    return {name: sources[f"analysis/learning_rule_inference/{name}"] for name in SCIENCE_SOURCES}


def collect(paths, configuration_hash, configuration, verify_arrays=True):
    groups, signatures, manifests = {}, [], []
    loss_selections = []
    seen_shards = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report["configuration_sha256"] != configuration_hash:
            raise ValueError(f"Protocol mismatch in {path}")
        identity = (report["profile_index"], report["split"], report["shard"])
        if identity in seen_shards:
            raise ValueError(f"Duplicate shard {identity}")
        seen_shards.add(identity)
        profile_index, split, shard = identity
        expected_start = configuration["splits"][split]["seed_start"] + 10000 * profile_index
        first = shard * configuration["shard_size"]
        last = min(first + configuration["shard_size"], configuration["splits"][split]["runs"])
        if report["seeds"] != list(range(expected_start + first, expected_start + last)):
            raise ValueError(f"Unexpected independent-run identities in {path}")
        labels = {"gd", "preconditioned"}
        labels.update(f"feedback-{m:g}" for m in configuration["feedback_mixtures"])
        labels.update(f"node-{r}" for r in configuration["node_repeats"])
        labels.update(f"node-shared-{r}" for r in configuration["shared_node_repeats"])
        expected_conditions = {(t, noise, label) for t in configuration["training_steps"]
                               for noise in configuration["response_noise"] for label in labels}
        actual_conditions = [(r["checkpoint_step"], r["response_noise"], r["rule"]) for r in report["results"]]
        if len(actual_conditions) != len(expected_conditions) or set(actual_conditions) != expected_conditions:
            raise ValueError(f"Incomplete or duplicated scientific conditions in {path}")
        if verify_arrays and digest(report["arrays"]) != report["arrays_sha256"]:
            raise ValueError(f"Array checksum mismatch in {path}")
        signatures.append(source_signature(report))
        manifests.append({"path": str(path), "sha256": digest(path),
                          "arrays": report["arrays"], "arrays_sha256": report["arrays_sha256"],
                          "job_id": report["provenance"]["slurm_job_id"],
                          "source_revision": report["provenance"]["analysis_commit"]})
        for result in report["results"]:
            base = (report["profile"]["name"], result["checkpoint_step"],
                    result["response_noise"], result["rule"], report["split"])
            for method in METHODS:
                key = (*base, method)
                group = groups.setdefault(key, {})
                for seed, value in zip(report["seeds"], result["scores"][method], strict=True):
                    if seed in group:
                        raise ValueError(f"Duplicate network seed {seed} in {key}")
                    group[seed] = math.nan if value is None else value
        if configuration.get("loss_matching_target"):
            reference = {r["checkpoint_step"]: r for r in report["results"]
                         if r["rule"] == "gd" and r["response_noise"] == 0}
            stages = sorted(reference)
            losses = np.asarray([reference[t]["loss"] for t in stages])
            target = configuration["loss_matching_target"]
            choice = np.abs(np.log(losses.clip(1e-30)) - math.log(target)).argmin(axis=0)
            chosen_loss = losses[choice, np.arange(len(report["seeds"]))]
            factor = configuration["loss_matching_factor"]
            eligible = (chosen_loss >= target / factor) & (chosen_loss <= target * factor)
            for index, seed in enumerate(report["seeds"]):
                loss_selections.append({"profile": report["profile"]["name"], "split": split,
                    "seed": seed, "selected_step": stages[choice[index]],
                    "selected_loss": float(chosen_loss[index]), "eligible": bool(eligible[index])})
            for noise in configuration["response_noise"]:
                for label in labels:
                    bank = {r["checkpoint_step"]: r for r in report["results"]
                            if r["rule"] == label and r["response_noise"] == noise}
                    for method in METHODS:
                        key = (report["profile"]["name"], -1, noise, label, split, method)
                        group = groups.setdefault(key, {})
                        scores = np.asarray([[math.nan if v is None else v for v in bank[t]["scores"][method]]
                                             for t in stages])
                        for index, seed in enumerate(report["seeds"]):
                            group[seed] = float(scores[choice[index], index]) if eligible[index] else math.nan
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Scientific source hashes differ across shards")
    # Confirm that all prescribed task/split/shard identities exist.
    for profile_index in range(len(configuration["profiles"])):
        for split in ("calibration", "evaluation"):
            n = configuration["splits"][split]["runs"]
            for shard in range(math.ceil(n / configuration["shard_size"])):
                if (profile_index, split, shard) not in seen_shards:
                    raise ValueError(f"Missing {profile_index=} {split=} {shard=}")
    for key, values in groups.items():
        expected = configuration["splits"][key[-2]]["runs"]
        if len(values) != expected:
            raise ValueError(f"Incomplete independent-run group {key}: {len(values)} != {expected}")
    return groups, manifests, signatures[0], loss_selections


def evaluate(groups, alpha=.05):
    rows, decisions = [], {}
    conditions = sorted({key[:3] for key in groups})
    for profile, step, noise in conditions:
        for method in METHODS:
            cutoff = max(conformal_cutoff(list(groups[(profile, step, noise, null, "calibration", method)].values()), alpha)
                         for null in NULLS)
            rules = sorted({key[3] for key in groups if key[:3] == (profile, step, noise)})
            for rule in rules:
                samples = groups[(profile, step, noise, rule, "evaluation", method)]
                valid = {seed: value for seed, value in samples.items() if math.isfinite(value)}
                rejected = {seed: value > cutoff for seed, value in valid.items()}
                count, rejections = len(valid), sum(rejected.values())
                lower, upper = wilson(rejections, count)
                rows.append({"profile": profile, "checkpoint_step": step, "response_noise": noise,
                    "method": method, "rule": rule, "cutoff": cutoff,
                    "evaluation_runs": count, "undefined_runs": len(samples) - count,
                    "rejections": rejections, "rejection_rate": rejections / count if count else math.nan,
                    "ci_low": lower, "ci_high": upper,
                    "role": "false_rejection" if rule in NULLS else "alternative_or_equivalence_control",
                    "calibration_gd_runs": sum(math.isfinite(x) for x in groups[(profile, step, noise, "gd", "calibration", method)].values()),
                    "calibration_preconditioned_runs": sum(math.isfinite(x) for x in groups[(profile, step, noise, "preconditioned", "calibration", method)].values())})
                decisions[(profile, step, noise, rule, method)] = rejected
    comparisons = []
    rng = np.random.default_rng(84602)
    for key, operator in decisions.items():
        profile, step, noise, rule, method = key
        if method != "operator" or rule not in ("feedback-0.1", "node-512"):
            continue
        for baseline in ("cosine", "pearson", "downhill", "population"):
            other = decisions[(profile, step, noise, rule, baseline)]
            seeds = sorted(set(operator) & set(other))
            differences = np.asarray([int(operator[s]) - int(other[s]) for s in seeds])
            if not len(differences):
                continue
            bootstrap = differences[rng.integers(len(seeds), size=(2000, len(seeds)))].mean(-1)
            comparisons.append({"profile": profile, "checkpoint_step": step, "response_noise": noise,
                "rule": rule, "baseline": baseline, "paired_runs": len(seeds),
                "power_difference": float(differences.mean()),
                "ci_low": float(np.quantile(bootstrap, .025)),
                "ci_high": float(np.quantile(bootstrap, .975))})
    return rows, comparisons


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/data/inference_evaluation.csv"))
    parser.add_argument("--paired-output", type=Path, default=Path("results/data/inference_paired_comparison.csv"))
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    if not configuration["locked"]:
        raise ValueError("Confirmatory summaries require a locked protocol")
    paths = sorted(args.input_dir.glob("*.json"))
    groups, manifests, sources, loss_selections = collect(paths, digest(args.configuration), configuration)
    rows, comparisons = evaluate(groups, configuration["alpha"])
    write_csv(args.output, rows)
    write_csv(args.paired_output, comparisons)
    if loss_selections:
        write_csv(args.output.with_name("inference_loss_matched_stages.csv"), loss_selections)
    save_json(args.output.with_suffix(".provenance.json"), {
        "protocol": configuration, "configuration_sha256": digest(args.configuration),
        "science_sources": sources, "completed_shards": manifests,
        "confidence_intervals": "Wilson 95% intervals for independent held-out runs, conditional on calibrated cutoffs",
        "paired_intervals": "2000 bootstrap resamples of paired independent-run decisions",
        "null_calibration": "Maximum of separate finite-sample split-conformal cutoffs for GD and fixed PSD-preconditioned GD",
    })
    print(f"Verified {len(paths)} completed shards; saved {len(rows)} held-out comparisons")


if __name__ == "__main__":
    main()
