"""Matched, independently seeded calibration and evaluation campaigns.

The simulation knows the update rule; scoring receives only G and V. Feedback
matrices are aligned once at initialization and remain fixed during training and
all restored-state probes. Magnitude scales use a separate target bank and one
positive scalar per network, checkpoint, and update rule.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import core, diagnostics, operators
from .provenance import ROOT, digest, provenance, save_json


def align_feedback(state, mixture):
    if not 0 <= mixture <= 1:
        raise ValueError("Feedback mixture must be between zero and one")
    def mix(true, random):
        scale = (torch.linalg.vector_norm(true, dim=(-2, -1), keepdim=True)
                 / torch.linalg.vector_norm(random, dim=(-2, -1), keepdim=True).clamp_min(1e-30))
        return (1 - mixture) * true + mixture * scale * random
    return replace(state,
        feedback=[mix(w[..., :-1].transpose(-1, -2), b)
                  for w, b in zip(state.weights, state.feedback)],
        readout_feedback=mix(state.decoder.transpose(-1, -2), state.readout_feedback))


def magnitude_errors(seeds, outputs, count=16):
    values = []
    for seed in seeds:
        rng = np.random.default_rng([int(seed), 4])
        error = rng.normal(size=(count, outputs))
        values.append(error / np.linalg.norm(error, axis=-1, keepdims=True))
    return torch.from_numpy(np.stack(values))


def direction_norm(updates):
    return sum(u.square().sum(dim=(-2, -1)) for u in updates).sqrt()


def matched_observation(state, x, errors, norm_errors, rule, eta, repeats=1,
                        sigma=1e-4, shared_noise=False, seed=0):
    norm_x = x[:, :1].expand(-1, norm_errors.shape[1], -1)
    reference = core.directions(state, norm_x, norm_errors, "gd")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    measured = core.directions(state, norm_x, norm_errors, rule, repeats,
                               sigma, generator, shared_noise)
    reference_rms = direction_norm(reference).square().mean(-1).sqrt()
    measured_rms = direction_norm(measured).square().mean(-1).sqrt()
    informative = (measured_rms > 1e-20) & (reference_rms > 1e-20)
    scale = torch.where(informative,
                        reference_rms / measured_rms.clamp_min(1e-30), 1.)
    del reference, measured
    # Independent random draws for scale estimation and the observed target bank.
    generator.manual_seed(seed + 1)
    updates = core.directions(state, x, errors, rule, repeats, sigma, generator, shared_noise)
    updates = [u * scale[:, None, None, None] for u in updates]
    _, before, zs = core.forward(state, x)
    _, after, new_zs = core.forward(state, x, core.apply_step(state, updates, eta))
    # The observed-layer gradient needs only the known decoder and output error.
    g = core.matvec(state.decoder.transpose(-1, -2), errors)
    v = -(after[-1] - before[-1]) / eta
    gate_changes = sum(((old > 0) != (new > 0)).sum(-1)
                       for old, new in zip(zs, new_zs))
    return g, v, {"magnitude_scale": scale, "scale_reference_rms": reference_rms,
                  "scale_rule_rms": measured_rms,
                  "magnitude_informative": informative,
                  "probe_direction_rms": direction_norm(updates).square().mean(-1).sqrt(),
                  "gate_changes": gate_changes.sum(-1)}


def score_observation(g, v, fit_count):
    gm, vm = g.transpose(-1, -2), v.transpose(-1, -2)
    compatibility = operators.compatibility(gm, vm)
    primary = torch.stack([compatibility[name]
                           for name in ("reciprocity", "negative_quadratic", "linearity")]).amax(0)
    # Retaining only +/- one direction isolates the information in independent
    # interventions while keeping the state, readout, and observations fixed.
    indices = [0, fit_count // 2]
    restricted = operators.compatibility(gm[..., indices], vm[..., indices])
    single_direction = torch.stack([restricted[name]
        for name in ("reciprocity", "negative_quadratic", "linearity")]).amax(0)
    alignment = diagnostics.alignment(g, v)
    reduced = {name: value.to(torch.float64).nanmedian(-1).values
               for name, value in alignment.items() if value.ndim == 2}
    population = (g * v).sum(dim=(-2, -1))
    denominator = (g.abs() * v.abs()).sum(dim=(-2, -1)).clamp_min(1e-30)
    baseline = {
        "cosine": 1 - reduced["cosine"],
        "pearson": 1 - reduced["pearson"],
        "downhill": 1 - reduced["downhill_fraction"],
        "population": (-population).clamp_min(0) / denominator,
    }
    predictions = operators.prediction_errors(gm[..., :fit_count], vm[..., :fit_count],
                                               gm[..., fit_count:], vm[..., fit_count:])
    return {"scores": {"operator": primary, "single_direction": single_direction, **baseline, **compatibility},
            "prediction": predictions, "alignment": reduced}


def tensor_state(state):
    return {key: ([w.detach().cpu() for w in value] if isinstance(value, list)
                  and value and isinstance(value[0], torch.Tensor)
                  else value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in asdict(state).items()}


def mask_failed(value, valid):
    if isinstance(value, dict):
        return {key: mask_failed(item, valid) for key, item in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(valid):
        shape = (len(valid),) + (1,) * (value.ndim - 1)
        if value.dtype == torch.bool:
            return value & valid.reshape(shape)
        return torch.where(valid.reshape(shape), value.to(torch.float64), torch.nan)
    return value


def rule_conditions(configuration):
    conditions = [("gd", "gd", 1, False, None),
                  ("preconditioned", "preconditioned", 1, False, None)]
    for mixture in configuration["feedback_mixtures"]:
        conditions.append((f"feedback-{mixture:g}", "feedback", 1, False, mixture))
    for repeats in configuration["node_repeats"]:
        conditions.append((f"node-{repeats}", "node", repeats, False, None))
    for repeats in configuration.get("shared_node_repeats", [1]):
        conditions.append((f"node-shared-{repeats}", "node", repeats, True, None))
    return conditions


def run(configuration, profile_index, split, shard, device, output):
    profile = configuration["profiles"][profile_index]
    split_config = configuration["splits"][split]
    start_index = shard * configuration["shard_size"]
    stop_index = min(start_index + configuration["shard_size"], split_config["runs"])
    if start_index >= stop_index:
        raise ValueError("Shard is outside the configured seed range")
    start_seed = split_config["seed_start"] + 10000 * profile_index
    seeds = list(range(start_seed + start_index, start_seed + stop_index))
    initial = core.initialize(seeds, profile["width"], profile["depth"], profile["outputs"]).to(device)
    data, targets = core.bar_data(seeds, profile["outputs"])
    data, targets = data.to(device), targets.to(device)
    errors, fit_count = core.probe_errors(seeds, profile["outputs"], configuration["heldout_probes"])
    errors = errors.to(device)
    norms = magnitude_errors(seeds, profile["outputs"], configuration["magnitude_probes"]).to(device)
    x = data[:, :1].expand(-1, errors.shape[1], -1)
    feedback = {mixture: align_feedback(initial, mixture)
                for mixture in configuration["feedback_mixtures"]}
    train_rule = profile.get("training_rule", "gd")
    if train_rule == "feedback":
        current = align_feedback(initial, profile.get("training_feedback_mixture", .1))
    else:
        current = initial
    all_results, observations, checkpoints, learning_curve = [], [], [], []
    previous_steps = 0
    for steps in configuration["training_steps"]:
        if steps > previous_steps:
            current, history = core.train(
                current, data, targets, steps - previous_steps,
                configuration["training_rates"][train_rule], rule=train_rule,
                repeats=configuration["training_node_repeats"], sigma=configuration["sigma"],
                seed=start_seed + shard * 100 + steps + 700,
            )
            learning_curve.extend([{"step": previous_steps + t, "loss": values} for t, values in history])
        previous_steps = steps
        loss = .5 * (core.forward(current, data)[0] - targets).square().sum(-1).mean(-1)
        valid = torch.isfinite(loss)
        # Preserve failed checkpoints and seeds. Mask them from inference rather
        # than sending NaNs into a batched eigensolver or silently dropping a run.
        computational_state = replace(current, weights=[torch.where(valid[:, None, None], w, 0.)
                                                         for w in current.weights])
        checkpoints.append({"step": steps, "state": tensor_state(current), "loss": loss.cpu()})
        reference_v = None
        for label, rule, repeats, shared, mixture in rule_conditions(configuration):
            if mixture is not None:
                variant = feedback[mixture]
                probe_state = replace(computational_state, feedback=variant.feedback,
                                      readout_feedback=variant.readout_feedback)
            else:
                probe_state = computational_state
            g, v, controls = matched_observation(
                probe_state, x, errors, norms, rule, configuration["eta"], repeats,
                configuration["sigma"], shared, seed=start_seed + shard * 10 + steps + 5000,
            )
            if label == "gd":
                reference_v = v.square().mean(dim=(-2, -1)).sqrt()
            for noise in configuration["response_noise"]:
                observed = v
                if noise:
                    generator = torch.Generator(device=device).manual_seed(start_seed + shard * 10 + steps + 9000)
                    observed = v + noise * reference_v[:, None, None] * torch.randn(
                        v.shape, dtype=v.dtype, device=v.device, generator=generator)
                result = score_observation(g, observed, fit_count)
                result = mask_failed(result, valid)
                common = {"checkpoint_step": steps, "rule": label, "response_noise": noise}
                all_results.append({**common, **result, "controls": mask_failed(controls, valid),
                                    "training_failed": ~valid,
                                    "loss": loss, "response_rms": mask_failed(v.square().mean(dim=(-2, -1)).sqrt(), valid)})
                observations.append({**common, "g": mask_failed(g, valid).cpu(),
                                      "v": mask_failed(observed, valid).cpu()})
            print(f"Completed {profile['name']} {split} shard={shard} step={steps} {label}", flush=True)
    arrays = output.with_suffix(".pt")
    arrays.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays.with_suffix(".pt.tmp")
    torch.save({"initial": tensor_state(initial), "checkpoints": checkpoints,
                "inputs": data[:, :1].cpu(), "probe_errors": errors.cpu(),
                "magnitude_errors": norms.cpu(), "observations": observations}, temporary)
    temporary.replace(arrays)
    return {"profile": profile, "profile_index": profile_index, "split": split, "shard": shard,
            "seeds": seeds, "results": all_results, "learning_curve": learning_curve,
            "arrays": str(arrays), "arrays_sha256": digest(arrays)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--profile", type=int, default=0)
    parser.add_argument("--split", choices=("development", "calibration", "evaluation"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.configuration.is_absolute() and (ROOT / args.configuration).exists():
        args.configuration = ROOT / args.configuration
    configuration_hash = digest(args.configuration)
    configuration = json.loads(args.configuration.read_text())
    if args.task_index is not None:
        if args.output_dir is None:
            parser.error("--task-index requires --output-dir")
        sizes = [math.ceil(configuration["splits"][s]["runs"] / configuration["shard_size"])
                 for s in ("calibration", "evaluation")]
        profile_tasks = sum(sizes)
        args.profile, local = divmod(args.task_index, profile_tasks)
        if args.profile >= len(configuration["profiles"]) or args.task_index < 0:
            parser.error("Array task is outside the protocol")
        args.split = "calibration" if local < sizes[0] else "evaluation"
        args.shard = local if local < sizes[0] else local - sizes[0]
        name = configuration["profiles"][args.profile]["name"]
        args.output = args.output_dir / f"{name}-{args.split}-s{args.shard:02}.json"
    if args.output is None or args.split is None:
        parser.error("Provide --output and --split, or an array task and output directory")
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing["configuration_sha256"] == configuration_hash and Path(existing["arrays"]).exists():
            if digest(existing["arrays"]) == existing["arrays_sha256"]:
                print(f"Verified existing completed shard {args.output}")
                return
        raise RuntimeError("Existing result does not match this protocol or arrays")
    if args.split != "development" and not configuration.get("locked"):
        raise ValueError("Calibration and evaluation require a locked protocol")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    metadata = provenance(args.device)
    start = time.perf_counter()
    result = run(configuration, args.profile, args.split, args.shard, args.device, args.output)
    final_metadata = provenance(args.device)
    if metadata["sources"] != final_metadata["sources"]:
        raise RuntimeError("Scientific implementation changed during this shard")
    if digest(args.configuration) != configuration_hash:
        raise RuntimeError("Protocol configuration changed during this shard")
    save_json(args.output, {"configuration": configuration, "configuration_path": str(args.configuration),
        "configuration_sha256": configuration_hash, "provenance": metadata,
        "peak_allocated_bytes": final_metadata.get("peak_allocated_bytes"),
        "seconds": time.perf_counter() - start, **result})
    print(f"Saved completed shard {args.output}", flush=True)


if __name__ == "__main__":
    main()
