"""Monte Carlo uncertainty in conditional mean activity responses.

The directional-gradient null uses exact gradients only in the simulator. The
studentized reciprocity statistic receives observed gradients, sample responses,
and their sample moments. Its cutoffs are calibrated on independent runs.
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch

from . import campaign, core, operators
from .provenance import ROOT, digest, provenance, save_json

RULES = ("directional-null", "preconditioned-null", "node", "feedback-0.1", "feedback-1")


def preparations(state, initial, x, errors):
    output, acts, zs = core.forward(state, x)
    exact = [u[..., -1] for u in core.directions(state, x, errors, "gd")]
    feedback = {}
    for mixture in (.1, 1.):
        fixed = campaign.align_feedback(initial, mixture)
        variant = replace(state, feedback=fixed.feedback, readout_feedback=fixed.readout_feedback)
        feedback[mixture] = [u[..., -1] for u in core.directions(variant, x, errors, "feedback")]
    return {"output": output, "activities": acts, "zs": zs, "exact": exact, "feedback": feedback}


def sample_signals(state, x, errors, prepared, count, sigma, generator):
    """Coupled physical node and exactly unbiased directional estimators."""
    node, directional = [], []
    target = (prepared["output"] - errors).unsqueeze(-2)
    xr = x.unsqueeze(-2).expand(*x.shape[:-1], count, x.shape[-1])
    for layer, delta in enumerate(prepared["exact"]):
        n = delta.shape[-1]
        eps = sigma * torch.randn((*x.shape[:-1], count, n), dtype=x.dtype,
                                   device=x.device, generator=generator)
        length_sq = eps.square().sum(-1, keepdim=True)
        exact = n * eps * (eps * delta.unsqueeze(-2)).sum(-1, keepdim=True) / length_sq
        noises = [None] * len(state.weights)
        noises[layer] = eps
        plus = core.forward(state, xr, noise=noises)[0]
        noises[layer] = -eps
        minus = core.forward(state, xr, noise=noises)[0]
        loss_difference = .5 * ((plus - target).square().sum(-1) - (minus - target).square().sum(-1))
        physical = n * loss_difference.unsqueeze(-1) * eps / (2 * length_sq)
        directional.append(exact)
        node.append(physical)
    result = {"directional-null": directional, "preconditioned-null": directional, "node": node}
    for mixture in (.1, 1.):
        result[f"feedback-{mixture:g}"] = [
            sample - gd.unsqueeze(-2) + fb.unsqueeze(-2)
            for sample, gd, fb in zip(directional, prepared["exact"], prepared["feedback"][mixture])]
    return result


def factored_responses(state, x, prepared, signals, eta, preconditioned=False):
    """Exact restored-step responses without materializing a weight set per draw.

    An update has rows signal_i * a_clean^T. Its action on an updated presynaptic
    vector is signal_i times their inner product. Elementwise fixed parameter
    preconditioning replaces that inner product with a row-weighted one.
    """
    count = signals[0].shape[-2]
    changed = x.unsqueeze(-2).expand(*x.shape[:-1], count, x.shape[-1])
    for layer, (w, signal) in enumerate(zip(state.weights, signals)):
        old_a = prepared["activities"][layer].unsqueeze(-2)
        products = torch.cat((old_a * changed, torch.ones_like(changed[..., :1])), -1)
        inner = (core.matvec(state.preconditioner[layer], products) if preconditioned
                 else products.sum(-1, keepdim=True))
        base = core.matvec(w[..., :-1], changed) + w[..., -1][:, None, None, :]
        changed = (base - eta * signal * inner).clamp_min(0)
    return -(changed - prepared["activities"][-1].unsqueeze(-2)) / eta


def contrasts(g, responses):
    """One antisymmetric response matrix per independent Monte Carlo draw."""
    # g: B,P,N; responses: B,P,R,N -> B,R,P,P.
    matrices = g.unsqueeze(1) @ responses.permute(0, 2, 3, 1)
    antisymmetric = matrices - matrices.transpose(-1, -2)
    indices = torch.triu_indices(g.shape[1], g.shape[1], offset=1, device=g.device)
    return antisymmetric[..., indices[0], indices[1]]


def mean_scores(g, v, contrast_sum, contrast_square_sum, response_sum,
                response_square_sum, count, reference):
    variance = ((contrast_square_sum - contrast_sum.square() / count) / (count - 1)).clamp_min(0)
    sem = (variance / count).sqrt()
    d = contrasts(g, v.unsqueeze(-2)).squeeze(1)
    floor = 1e-12 * d.abs().amax(-1, keepdim=True).clamp_min(1.)
    statistic = (d.abs() / sem.clamp_min(floor)).amax(-1)
    c = operators.compatibility(g.transpose(-1, -2), v.transpose(-1, -2))
    raw = torch.stack([c[key] for key in ("reciprocity", "negative_quadratic", "linearity")]).amax(0)
    response_variance = ((response_square_sum - response_sum.square() / count) / (count - 1)).clamp_min(0)
    reference_norm = operators.norm(reference.transpose(-1, -2)).clamp_min(1e-30)
    error = operators.norm((v - reference).transpose(-1, -2)) / reference_norm
    mc_scale = (response_variance.sum(dim=(-2, -1)) / count).sqrt() / reference_norm
    return {"studentized_reciprocity": statistic, "raw_operator": raw,
            "relative_error_to_deterministic_mean_step": error,
            "mc_variability_scale": mc_scale, "degenerate_contrasts": (sem < floor).sum(-1)}


def simulate(state, initial, x, errors, configuration, seed):
    prepared = preparations(state, initial, x, errors)
    g = core.matvec(state.decoder.transpose(-1, -2), errors)
    references = {}
    for name in RULES:
        exact = prepared["exact"] if not name.startswith("feedback") else prepared["feedback"][float(name.split("-")[1])]
        references[name] = factored_responses(state, x, prepared, [s.unsqueeze(-2) for s in exact],
                                             configuration["eta"], name == "preconditioned-null").squeeze(-2)
    # Deterministic known-null scores calibrate the deliberately naive comparator.
    deterministic = {}
    for name in ("directional-null", "preconditioned-null"):
        c = operators.compatibility(g.transpose(-1, -2), references[name].transpose(-1, -2))
        deterministic[name] = torch.stack([c[k] for k in ("reciprocity", "negative_quadratic", "linearity")]).amax(0)
    accumulators = {name: None for name in RULES}
    generator = torch.Generator(device=x.device).manual_seed(seed)
    completed = 0
    rows, arrays = [], []
    for target_count in configuration["sample_counts"]:
        while completed < target_count:
            count = min(configuration["chunk_size"], target_count - completed)
            signals = sample_signals(state, x, errors, prepared, count, configuration["sigma"], generator)
            for name, signal in signals.items():
                v = factored_responses(state, x, prepared, signal, configuration["eta"], name == "preconditioned-null")
                d = contrasts(g, v)
                sums = {"v": v.sum(-2), "v2": v.square().sum(-2),
                        "d": d.sum(-2), "d2": d.square().sum(-2),
                        "signals": [s.sum(-2) for s in signal]}
                previous = accumulators[name]
                if previous is None:
                    accumulators[name] = sums
                else:
                    for key in ("v", "v2", "d", "d2"):
                        previous[key] += sums[key]
                    for l, value in enumerate(sums["signals"]):
                        previous["signals"][l] += value
            completed += count
        for name, sums in accumulators.items():
            averaged_responses = sums["v"] / completed
            mean_signals = [s.unsqueeze(-2) / completed for s in sums["signals"]]
            averaged_directions = factored_responses(state, x, prepared, mean_signals, configuration["eta"],
                                                     name == "preconditioned-null").squeeze(-2)
            gap = operators.norm((averaged_responses - averaged_directions).transpose(-1, -2))
            reference_norm = operators.norm(references[name].transpose(-1, -2)).clamp_min(1e-30)
            for mode, mean_v in (("responses-first", averaged_responses), ("directions-first", averaged_directions)):
                statistics = mean_scores(g, mean_v, sums["d"], sums["d2"], sums["v"], sums["v2"],
                                         completed, references[name])
                common = {"rule": name, "mode": mode, "samples": completed}
                rows.append({**common, **statistics, "mean_order_gap": gap / reference_norm})
                arrays.append({**common, "g": g.cpu(), "v": mean_v.cpu(),
                               "contrast_sum": sums["d"].cpu().clone(),
                               "contrast_square_sum": sums["d2"].cpu().clone()})
        print(f"Completed conditional means for {len(state.seeds)} runs at R={completed}", flush=True)
    return rows, arrays, deterministic


def slice_state(values, start, stop, device):
    state = core.State(**values)
    return core.State([w[start:stop] for w in state.weights], state.decoder[start:stop],
        [b[start:stop] for b in state.feedback], state.readout_feedback[start:stop],
        [p[start:stop] for p in state.preconditioner], state.seeds[start:stop]).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config_path = ROOT / args.configuration if not args.configuration.is_absolute() else args.configuration
    configuration = json.loads(config_path.read_text())
    shard_count = configuration["shards_per_split"]
    if not 0 <= args.task_index < 2 * shard_count:
        parser.error("Mean-response array task is outside the configured split/shard range")
    split = "calibration" if args.task_index < shard_count else "evaluation"
    shard = args.task_index % shard_count
    parent_path = args.input_dir / f"primary-{split}-s{shard:02}.json"
    parent = json.loads(parent_path.read_text())
    if parent["configuration"]["protocol"] != configuration["parent_protocol"] or parent["profile"]["name"] != "primary":
        raise RuntimeError("Unexpected source checkpoint protocol or profile")
    if digest(parent["arrays"]) != parent["arrays_sha256"]:
        raise RuntimeError("Parent checkpoint arrays failed verification")
    output = args.output_dir / f"primary-{split}-s{shard:02}.json"
    if output.exists():
        old = json.loads(output.read_text())
        if old["configuration_sha256"] == digest(config_path) and digest(old["arrays"]) == old["arrays_sha256"]:
            print(f"Verified completed conditional-mean shard {output}")
            return
        raise RuntimeError("Existing conditional-mean result is inconsistent")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    metadata = provenance(args.device)
    start_time = time.perf_counter()
    data = torch.load(parent["arrays"], map_location="cpu", weights_only=True, mmap=True)
    checkpoint = next(c for c in data["checkpoints"] if c["step"] == configuration["checkpoint_step"])
    batch = args.batch_size or (2 if args.device == "cpu" else len(parent["seeds"]))
    combined, arrays, deterministic = {}, [], {}
    for first in range(0, len(parent["seeds"]), batch):
        last = min(first + batch, len(parent["seeds"]))
        state = slice_state(checkpoint["state"], first, last, args.device)
        initial = slice_state(data["initial"], first, last, args.device)
        errors = data["probe_errors"][first:last].to(args.device)
        x = data["inputs"][first:last].to(args.device).expand(-1, errors.shape[1], -1)
        rows, raw, controls = simulate(state, initial, x, errors, configuration,
                                      seed=parent["seeds"][first] + 6000)
        for row in rows:
            key = (row["rule"], row["mode"], row["samples"])
            stored = combined.setdefault(key, {"rule": key[0], "mode": key[1], "samples": key[2]})
            for name, value in row.items():
                if isinstance(value, torch.Tensor):
                    stored.setdefault(name, []).extend(value.cpu().tolist())
        arrays.append({"seeds": state.seeds, "observations": raw})
        for name, value in controls.items():
            deterministic.setdefault(name, []).extend(value.cpu().tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".pt")
    temporary = raw_path.with_suffix(".pt.tmp")
    torch.save(arrays, temporary)
    temporary.replace(raw_path)
    final_metadata = provenance(args.device)
    if metadata["sources"] != final_metadata["sources"]:
        raise RuntimeError("Mean-response implementation changed during the experiment")
    save_json(output, {"configuration": configuration, "configuration_sha256": digest(config_path),
        "parent": str(parent_path), "parent_sha256": digest(parent_path), "split": split, "shard": shard,
        "seeds": parent["seeds"], "batch_size": batch, "provenance": metadata,
        "peak_allocated_bytes": final_metadata.get("peak_allocated_bytes"),
        "seconds": time.perf_counter() - start_time, "results": list(combined.values()),
        "deterministic_null_scores": deterministic,
        "arrays": str(raw_path), "arrays_sha256": digest(raw_path)})
    print(f"Saved conditional-mean shard {output}", flush=True)


if __name__ == "__main__":
    main()
