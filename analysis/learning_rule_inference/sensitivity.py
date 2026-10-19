"""Exploratory finite-step and perturbation-size checks on development checkpoints."""

import argparse
import json
from pathlib import Path
import time

import torch

from . import campaign, core, operators
from .mean_response import slice_state
from .provenance import digest, provenance, save_json


def score(g, v):
    c = operators.compatibility(g.transpose(-1, -2), v.transpose(-1, -2))
    return torch.stack([c[k] for k in ("reciprocity", "negative_quadratic", "linearity")]).amax(0)


def shared_psd_updates(state, x, errors, norm_errors, repeats, seed):
    """A fixed symmetric PSD transformation of the exact parameter gradient."""
    generator = torch.Generator(device=x.device).manual_seed(seed)
    update = core.directions(state, x, errors)
    norm_x = x[:, :1].expand(-1, norm_errors.shape[1], -1)
    norm_update = core.directions(state, norm_x, norm_errors)
    measured, norm_measured = [], []
    for u, nu in zip(update, norm_update):
        width = u.shape[-2]
        directions = torch.randn((len(state.seeds), repeats, width), dtype=x.dtype,
                                 device=x.device, generator=generator)
        directions /= directions.norm(dim=-1, keepdim=True)
        h = width * directions.transpose(-1, -2) @ directions / repeats
        measured.append(torch.einsum("bij,bpjk->bpik", h, u))
        norm_measured.append(torch.einsum("bij,bpjk->bpik", h, nu))
    reference = campaign.direction_norm(norm_update).square().mean(-1).sqrt()
    scale = reference / campaign.direction_norm(norm_measured).square().mean(-1).sqrt().clamp_min(1e-30)
    return [u * scale[:, None, None, None] for u in measured]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/generated/inference/matched-development-p0.json"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/inference/sensitivity-development.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = provenance(args.device)
    start = time.perf_counter()
    parent = json.loads(args.checkpoint.read_text())
    if parent["split"] != "development" or digest(parent["arrays"]) != parent["arrays_sha256"]:
        raise ValueError("Sensitivity checks require a verified development checkpoint")
    data = torch.load(parent["arrays"], weights_only=True, map_location="cpu")
    checkpoint = next(c for c in data["checkpoints"] if c["step"] == 1000)
    state = slice_state(checkpoint["state"], 0, len(parent["seeds"]), args.device)
    errors = data["probe_errors"].to(args.device)
    norm_errors = data["magnitude_errors"].to(args.device)
    x = data["inputs"].to(args.device).expand(-1, errors.shape[1], -1)
    _, before, zs = core.forward(state, x)
    g = core.matvec(state.decoder.transpose(-1, -2), errors)
    rows, arrays = [], []
    for repeats in (1, 512):
        for variant, sigma in (("exact-shared-PSD", None), ("physical-node", 1e-4), ("physical-node", 1e-6)):
            if sigma is None:
                updates = shared_psd_updates(state, x, errors, norm_errors, repeats, 118000)
            else:
                norm_x = x[:, :1].expand(-1, norm_errors.shape[1], -1)
                reference = core.directions(state, norm_x, norm_errors)
                generator = torch.Generator(device=args.device).manual_seed(118000)
                scale_update = core.directions(state, norm_x, norm_errors, "node", repeats, sigma, generator, True)
                scale = campaign.direction_norm(reference).square().mean(-1).sqrt() / campaign.direction_norm(scale_update).square().mean(-1).sqrt().clamp_min(1e-30)
                generator.manual_seed(118001)
                updates = core.directions(state, x, errors, "node", repeats, sigma, generator, True)
                updates = [u * scale[:, None, None, None] for u in updates]
            first_order = core.activity_jvp(state, x, updates)[-1]
            for eta in (1e-3, 1e-4, 1e-5, 1e-6):
                _, after, new_zs = core.forward(state, x, core.apply_step(state, updates, eta))
                v = -(after[-1] - before[-1]) / eta
                relative = operators.norm((v - first_order).transpose(-1, -2)) / operators.norm(first_order.transpose(-1, -2)).clamp_min(1e-30)
                gates = sum(((z > 0) != (nz > 0)).sum(dim=(-2, -1)) for z, nz in zip(zs, new_zs))
                row = {"variant": variant, "sigma": sigma, "repeats": repeats, "eta": eta,
                       "operator": score(g, v), "first_order_operator": score(g, first_order),
                       "relative_first_order_error": relative, "gate_changes": gates}
                rows.append(row)
                arrays.append({"variant": variant, "sigma": sigma, "repeats": repeats, "eta": eta,
                               "g": g.cpu(), "v": v.cpu(), "first_order": first_order.cpu()})
                print(f"Sensitivity {variant} sigma={sigma} R={repeats} eta={eta}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".pt")
    torch.save(arrays, raw_path)
    if metadata["sources"] != provenance(args.device)["sources"]:
        raise RuntimeError("Sensitivity sources changed during execution")
    save_json(args.output, {"provenance": metadata, "parent": str(args.checkpoint),
        "parent_sha256": digest(args.checkpoint), "seeds": parent["seeds"], "results": rows,
        "arrays": str(raw_path), "arrays_sha256": digest(raw_path), "seconds": time.perf_counter() - start,
        "status": "Exploratory development-only sensitivity after the locked main evaluation"})


if __name__ == "__main__":
    main()
