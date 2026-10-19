"""Development-only synthetic layer losses and raw/matched scale comparisons.

These checks complete bounded controls from the research plan. They neither
change the locked main statistic nor choose cutoffs from evaluation networks.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import campaign, core, diagnostics, operators
from .mean_response import slice_state
from .provenance import ROOT, digest, provenance, save_json


def operator_score(g, v):
    scores = operators.compatibility(g.transpose(-1, -2), v.transpose(-1, -2))
    return torch.stack([scores[k] for k in ("reciprocity", "negative_quadratic", "linearity")]).amax(0)


def layer_loss_checks(state, x, target, config):
    output, clean, _ = core.forward(state, x)
    task_error = output - target
    task_g = core.matvec(state.decoder.transpose(-1, -2), task_error)
    width = clean[-1].shape[-1]
    random = np.stack([np.random.default_rng([seed, 19]).normal(size=(1, width)) for seed in state.seeds])
    synthetic = torch.from_numpy(random).to(x.device)
    synthetic = synthetic / synthetic.norm(dim=-1, keepdim=True) * task_g.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(synthetic.norm(dim=-1), task_g.norm(dim=-1), rtol=1e-12, atol=1e-14)
    identity = torch.eye(width, dtype=x.dtype, device=x.device).expand(len(state.seeds), -1, -1)
    hidden_loss_state = replace(state, decoder=identity)
    cases = {"task": (state, task_error), "synthetic-layer": (hidden_loss_state, synthetic)}
    rows, arrays = [], []
    for label in config["loss_controls"]:
        current, error = cases[label]
        old_output, old_acts, zs = core.forward(current, x)
        fixed_target = old_output - error
        gradients = core.activity_gradients(current, error, zs) + [error]
        updates = core.directions(current, x, error)
        for mask in config["parameter_masks"]:
            layers = {"all": None, "own": [len(state.weights)-1],
                      "upstream": list(range(len(state.weights)-1))}[mask]
            kernels = core.kernels(current, x, layers=layers)
            decoder = current.decoder.unsqueeze(1)
            kernels.append(decoder @ kernels[-1] @ decoder.transpose(-1, -2))
            for eta in config["etas"]:
                after_output, after_acts, new_zs = core.forward(current, x, core.apply_step(current, updates, eta, layers))
                actual_loss_change = .5*((after_output-fixed_target).square().sum(-1)-error.square().sum(-1))
                measured = diagnostics.step_diagnostics(current, x, error, eta, layers)
                for layer, record in enumerate(measured):
                    common = {"loss_control": label, "parameter_mask": mask, "eta": eta,
                              "layer": layer, "layer_type": record["layer_type"]}
                    rows.append({**common, "actual_loss_change": actual_loss_change, **record})
                    old = old_acts[layer+1] if layer<len(state.weights) else old_output
                    new = after_acts[layer+1] if layer<len(state.weights) else after_output
                    g, k = gradients[layer], kernels[layer]
                    self_term = k.diagonal(dim1=-2, dim2=-1)*g
                    interference = core.matvec(k, g)-self_term
                    arrays.append({**common, "g": g.cpu(), "delta_activity": (new-old).cpu(),
                        "coordinate_loss_contribution": (g*(new-old)).cpu(),
                        "signed_self_term": self_term.cpu(), "signed_interference": interference.cpu(),
                        "gate_on": ((zs[layer]<=0)&(new_zs[layer]>0)).cpu() if layer<len(state.weights) else None,
                        "gate_off": ((zs[layer]>0)&(new_zs[layer]<=0)).cpu() if layer<len(state.weights) else None})
        print(f"Completed matched-state {label} mechanism control", flush=True)
    return rows, arrays


def scale_checks(state, initial, data, config):
    errors = data["probe_errors"].to(state.decoder.device)
    norm_errors = data["magnitude_errors"].to(state.decoder.device)
    x = data["inputs"].to(state.decoder.device).expand(-1, errors.shape[1], -1)
    before = core.forward(state, x)[1][-1]
    eta = config["raw_scale_eta"]
    sigma = config["raw_scale_sigma"]
    seed = state.seeds[0] + config["checkpoint_step"] + 5000
    conditions = campaign.rule_conditions(config)
    rows, arrays = [], []
    for label, rule, repeats, shared, mixture in conditions:
        if mixture is None:
            current = state
        else:
            feedback = campaign.align_feedback(initial, mixture)
            current = replace(state, feedback=feedback.feedback, readout_feedback=feedback.readout_feedback)
        g, matched, controls = campaign.matched_observation(current, x, errors, norm_errors,
            rule, eta, repeats, sigma, shared, seed)
        generator = torch.Generator(device=x.device).manual_seed(seed+1)
        updates = core.directions(current, x, errors, rule, repeats, sigma, generator, shared)
        raw_after = core.forward(current, x, core.apply_step(current, updates, eta))[1][-1]
        raw = -(raw_after-before)/eta
        scale = controls["magnitude_scale"]
        scaled_updates = [u*scale[:,None,None,None] for u in updates]
        replay = -(core.forward(current, x, core.apply_step(current, scaled_updates, eta))[1][-1]-before)/eta
        torch.testing.assert_close(replay, matched, rtol=1e-10, atol=1e-11)
        first = core.activity_jvp(current, x, updates)[-1]
        first_score = operator_score(g, first)
        first_scaled_score = operator_score(g, first*scale[:,None,None])
        if (first_score-first_scaled_score).abs().max()>1e-8:
            raise AssertionError("A constant positive bank scale changed the first-order compatibility score")
        common = {"rule":label, "eta":eta, "sigma":sigma, "magnitude_scale":scale,
                  "first_order_scale_invariance_error":(first_score-first_scaled_score).abs()}
        for mode, v, prediction in (("raw",raw,first), ("matched",matched,first*scale[:,None,None])):
            error = operators.norm((v-prediction).transpose(-1,-2))/operators.norm(prediction.transpose(-1,-2)).clamp_min(1e-30)
            rows.append({**common, "mode":mode, "operator":operator_score(g,v),
                         "first_order_operator":first_score, "relative_first_order_error":error,
                         "response_rms":v.square().mean(dim=(-2,-1)).sqrt()})
            arrays.append({"rule":label,"mode":mode,"g":g.cpu(),"v":v.cpu(),
                           "first_order_response":prediction.cpu(),"magnitude_scale":scale.cpu()})
        print(f"Verified same-draw raw/matched comparison for {label}",flush=True)
    return rows, arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration",type=Path,default=Path("configs/inference/mechanism_controls_v1.json"))
    parser.add_argument("--checkpoint",type=Path,default=Path("data/generated/inference/matched-development-p0.json"))
    parser.add_argument("--output",type=Path,default=Path("data/generated/inference/mechanism-controls-development.json"))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    config_path=ROOT/args.configuration if not args.configuration.is_absolute() else args.configuration
    config=json.loads(config_path.read_text())
    parent=json.loads(args.checkpoint.read_text())
    if parent["split"]!="development" or parent["profile"]["name"]!="primary":
        raise ValueError("Use only the primary development checkpoints for these exploratory checks")
    if digest(parent["arrays"])!=parent["arrays_sha256"]:
        raise ValueError("Parent development checkpoint failed verification")
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old["configuration_sha256"]==digest(config_path) and digest(old["arrays"])==old["arrays_sha256"]:
            print("Verified completed development mechanism controls")
            return
        raise ValueError("Existing development controls differ from this configuration")
    torch.set_num_threads(1)
    metadata=provenance(args.device)
    start=time.perf_counter()
    data=torch.load(parent["arrays"],weights_only=True,map_location="cpu")
    checkpoint=next(c for c in data["checkpoints"] if c["step"]==config["checkpoint_step"])
    count=len(parent["seeds"])
    state=slice_state(checkpoint["state"],0,count,args.device)
    initial=slice_state(data["initial"],0,count,args.device)
    inputs,targets=core.bar_data(state.seeds,parent["profile"]["outputs"])
    torch.testing.assert_close(inputs[:,:1],data["inputs"],rtol=0,atol=0)
    rows,layer_arrays=layer_loss_checks(state,inputs[:,:1].to(args.device),targets[:,:1].to(args.device),config)
    scales,scale_arrays=scale_checks(state,initial,data,config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw_path=args.output.with_suffix(".pt")
    torch.save({"layer_loss":layer_arrays,"scale":scale_arrays},raw_path)
    if metadata["sources"]!=provenance(args.device)["sources"]:
        raise RuntimeError("Scientific sources changed during development controls")
    save_json(args.output,{"configuration":config,"configuration_sha256":digest(config_path),
        "parent":str(args.checkpoint),"parent_sha256":digest(args.checkpoint),"seeds":parent["seeds"],
        "provenance":metadata,"seconds":time.perf_counter()-start,
        "layer_loss_results":rows,"scale_results":scales,
        "arrays":str(raw_path),"arrays_sha256":digest(raw_path)})
    print(f"Saved development mechanism and scale controls: {args.output}",flush=True)


if __name__=="__main__":
    main()
