"""Resumable CPU/CUDA development experiments and representative benchmarks."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from . import core, diagnostics, operators
from .provenance import digest, provenance, save_json


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def observe(state, x, errors, rule, eta, repeats, sigma, seed, shared_noise=False):
    _, before, zs = core.forward(state, x)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    updates = core.directions(state, x, errors, rule, repeats, sigma, generator, shared_noise)
    after = core.forward(state, x, core.apply_step(state, updates, eta))[1][-1]
    g = core.activity_gradients(state, errors, zs)[-1]
    v = -(after - before[-1]) / eta
    weight_norm = sum(u.square().sum(dim=(-2, -1)) for u in updates).sqrt()
    return g, v, weight_norm


def benchmark(args, state, data):
    output = []
    for dtype in (torch.float64, torch.float32):
        current = state.to(args.device, dtype)
        errors, _ = core.probe_errors(state.seeds, args.outputs)
        errors = errors.to(args.device, dtype)
        x = data[:, :1].to(args.device, dtype).expand(-1, errors.shape[1], -1)
        # Warm up library initialization and GPU kernels before measuring.
        for rule, repeats in (("gd", 1), ("feedback", 1), ("node", 32)):
            observe(current, x, errors, rule, args.eta, repeats, args.sigma, 41)
            synchronize(args.device)
            start = time.perf_counter()
            for _ in range(args.benchmark_repeats):
                observe(current, x, errors, rule, args.eta, repeats, args.sigma, 42)
            synchronize(args.device)
            seconds = (time.perf_counter() - start) / args.benchmark_repeats
            output.append({"dtype": str(dtype), "rule": rule, "perturbation_repeats": repeats,
                           "seconds_per_bank": seconds, "networks": len(state.seeds),
                           "probes_per_network": errors.shape[1],
                           "network_probes_per_second": len(state.seeds) * errors.shape[1] / seconds})
    return {"timings": output}


def experiment(args, state, data, targets):
    device = args.device
    current = state.to(device)
    data, targets = data.to(device), targets.to(device)
    history = []
    if args.train_steps:
        current, history = core.train(current, data, targets, args.train_steps,
                                      args.train_eta, seed=args.seed_start + 700)
    errors, fit_count = core.probe_errors(state.seeds, args.outputs)
    errors = errors.to(device)
    x = data[:, :1].expand(-1, errors.shape[1], -1)
    all_results, observations = [], []
    for eta in args.etas:
        for rule, repeats, shared in [("gd", 1, False), ("preconditioned", 1, False),
                                      ("feedback", 1, False)] + [
                ("node", r, common) for r in args.node_repeats for common in (False, True)]:
            g, v, norms = observe(current, x, errors, rule, eta, repeats, args.sigma,
                                  args.seed_start + 1000, shared)
            observations.append({"rule": rule, "eta": eta, "node_repeats": repeats,
                                 "shared_noise": shared, "g": g.cpu(), "v": v.cpu()})
            scores = operators.compatibility(g.transpose(-1, -2), v.transpose(-1, -2))
            predictions = operators.prediction_errors(
                g[:, :fit_count].transpose(-1, -2), v[:, :fit_count].transpose(-1, -2),
                g[:, fit_count:].transpose(-1, -2), v[:, fit_count:].transpose(-1, -2),
                iterations=args.fit_iterations,
            )
            all_results.append({"rule": rule, "eta": eta, "node_repeats": repeats,
                                "shared_noise": shared, "scores": scores,
                                "prediction": predictions,
                                "alignment": diagnostics.alignment(g, v), "weight_update_direction_norm": norms})
            print(f"Completed {rule} eta={eta:g} R={repeats} shared={shared}", flush=True)
    mechanisms = []
    for eta in args.etas:
        task_error = core.forward(current, data[:, :1])[0] - targets[:, :1]
        for name, layers in (("all", None), ("own", [args.depth - 1]),
                             ("upstream", list(range(args.depth - 1)))):
            mechanisms.append({"eta": eta, "parameter_mask": name,
                               "layers": diagnostics.step_diagnostics(current, data[:, :1], task_error, eta, layers)})
    arrays = args.output.with_suffix(".pt")
    temporary = arrays.with_suffix(".pt.tmp")
    arrays.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint": {key: ([w.detach().cpu() for w in value] if isinstance(value, list)
                                      and value and isinstance(value[0], torch.Tensor)
                                      else value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                                for key, value in asdict(current).items()},
                "inputs": data[:, :1].cpu(), "probe_errors": errors.cpu(),
                "task_targets": targets[:, :1].cpu(), "observations": observations}, temporary)
    temporary.replace(arrays)
    return {"results": all_results, "mechanisms": mechanisms, "training_history": history,
            "fit_probes": fit_count, "heldout_probes": errors.shape[1] - fit_count,
            "checkpoint_path": str(arrays), "checkpoint_sha256": digest(arrays)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "pilot"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--outputs", type=int, default=8)
    parser.add_argument("--networks", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--etas", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--node-repeats", type=int, nargs="+", default=[1, 32, 512])
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--train-eta", type=float, default=.005)
    parser.add_argument("--fit-iterations", type=int, default=500)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite completed result {args.output}")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    metadata = provenance(args.device)
    start = time.perf_counter()
    seeds = list(range(args.seed_start, args.seed_start + args.networks))
    state = core.initialize(seeds, args.width, args.depth, args.outputs)
    data, targets = core.bar_data(seeds, args.outputs)
    result = benchmark(args, state, data) if args.mode == "benchmark" else experiment(args, state, data, targets)
    synchronize(args.device)
    finished = provenance(args.device)
    if metadata["sources"] != finished["sources"] or metadata["lockfile_sha256"] != finished["lockfile_sha256"]:
        raise RuntimeError("Implementation or lockfile changed during the experiment; rerun from a fixed revision")
    save_json(args.output, {"configuration": {**vars(args), "output": str(args.output)},
                            "seeds": seeds, "seconds": time.perf_counter() - start,
                            "provenance": {**metadata, "peak_allocated_bytes": finished.get("peak_allocated_bytes")}, **result})
    print(f"Saved completed result {args.output}", flush=True)


if __name__ == "__main__":
    main()
