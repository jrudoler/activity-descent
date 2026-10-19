"""Bounded layerwise reanalysis of the original bar-classification simulation.

Load only named function definitions from the original figure script, avoiding
its top-level architecture sweeps. Retain the trainable linear output layer,
original data generator, RNG seeds, online updates, and active-neuron pooling.
New layerwise diagnostics are checked against the original sweep diagnostics.
"""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import torch

from . import diagnostics
from .provenance import ROOT, clean_json, digest, provenance, save_json


def original_functions():
    source = ROOT / "analysis/generate_main_figures/run.py"
    names = {"create_network", "forward", "backprop", "make_bar_images", "corr",
             "compute_layer_local_prediction", "compute_kernel_diagonal_predictions", "run_experiment"}
    definitions = [node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError("Original simulation entrypoints changed; review the continuity analysis")
    namespace = {"np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, source


def run(seed, width, depth, steps, eta, every):
    original, source = original_functions()
    sizes = [16] + [width] * depth + [2]
    weights = original["create_network"](sizes, np.random.default_rng(1000 + seed))
    x, targets = original["make_bar_images"](np.random.default_rng(2000 + seed))
    rng = np.random.default_rng(3000 + seed)
    rows, pooled = [], {"raw": [], "diagonal": [], "kernel": []}
    for step in range(steps):
        index = rng.integers(len(x))
        before, zs = original["forward"](weights, x[index])
        gradients, updates = original["backprop"](weights, before, zs, targets[index])
        measure = step % every == 0 or step == steps - 1
        if measure:
            kernels, active, previous = [], [], None
            for layer, w in enumerate(weights):
                gate = (zs[layer + 1] > 0).astype(float) if layer < depth else np.ones(2)
                local = np.diag((np.sum(before[layer] ** 2) + 1) * gate)
                if previous is not None:
                    b = gate[:, None] * w[:, :-1]
                    local += b @ previous @ b.T
                kernels.append(local)
                previous = local
                active.append(gate > 0)
            original_prediction, original_diagonal, original_raw, _, _ = original["compute_kernel_diagonal_predictions"](
                sizes, weights, before, zs, gradients, eta)
            full = np.concatenate([-eta * k @ g for k, g in zip(kernels, gradients[1:])])
            diagonal = np.concatenate([-eta * np.diag(k) * g for k, g in zip(kernels, gradients[1:])])
            np.testing.assert_allclose(full, original_prediction, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(diagonal, original_diagonal, atol=1e-10, rtol=1e-10)
        weights = [w - eta * u for w, u in zip(weights, updates)]
        if measure:
            after, new_zs = original["forward"](weights, x[index])
            changes = [a - b for a, b in zip(after[1:], before[1:])]
            mask = np.concatenate(active)
            flat_changes = np.concatenate(changes)[mask]
            pooled["raw"].append(original["corr"](flat_changes, original_raw[mask]))
            pooled["diagonal"].append(original["corr"](flat_changes, original_diagonal[mask]))
            pooled["kernel"].append(original["corr"](flat_changes, original_prediction[mask]))
            for layer, (g, k, change) in enumerate(zip(gradients[1:], kernels, changes)):
                t = lambda value: torch.from_numpy(np.asarray(value)).reshape(1, -1)
                alignment = {key: value[0].item() for key, value in diagnostics.alignment(t(g), t(-change / eta)).items()}
                own = np.diag(k) * g
                interference = k @ g - own
                norm_change = np.linalg.norm(change)
                row = {"seed": seed, "step": step, "layer": layer,
                       "layer_type": "hidden" if layer < depth else "output", **alignment,
                       "active_neurons": int(active[layer].sum()),
                       "gate_changes": int(np.sum((zs[layer + 1] > 0) != (new_zs[layer + 1] > 0))) if layer < depth else 0,
                       "full_relative_error": np.linalg.norm(change + eta * k @ g) / norm_change if norm_change else None,
                       "diagonal_relative_error": np.linalg.norm(change + eta * own) / norm_change if norm_change else None,
                       "self_l2": float(np.linalg.norm(own)),
                       "interference_l2": float(np.linalg.norm(interference)),
                       "gradient_dot_activity_change": float(g @ change)}
                rows.append(row)
    # Exact continuity check against the untouched original experiment loop.
    reference = original["run_experiment"](width, depth, eta, steps, every,
        full_jacobian=False, network_seed=1000 + seed, data_seed=2000 + seed, sampling_seed=3000 + seed)[0]
    for name, field in (("raw", "corr_raw_gradient"), ("diagonal", "corr_diagonal"), ("kernel", "corr_kernel")):
        np.testing.assert_allclose(pooled[name], reference[field], atol=1e-12, rtol=1e-12)
    return rows, {"seed": seed, "pooled": pooled, "original_history": reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", type=int, default=20)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=.005)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, histories = [], []
    for seed in range(args.networks):
        measured, history = run(seed, args.width, args.depth, args.steps, args.eta, args.every)
        rows.extend(measured)
        histories.append(history)
        print(f"Verified original simulation continuity for seed {seed}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, source = original_functions()
    save_json(args.output.with_suffix(".provenance.json"), {
        "configuration": {**vars(args), "output": str(args.output)},
        "original_source_sha256": digest(source), "original_source": str(source),
        "provenance": provenance("cpu"), "histories": histories,
        "continuity_check": "Layer kernels and all pooled correlations match the original experiment to 1e-12",
    })


if __name__ == "__main__":
    main()
