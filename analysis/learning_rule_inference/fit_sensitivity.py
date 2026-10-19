"""Sensitivity of supporting PSD fits to truncation tolerance and ridge bias.

Positive ridge gives a unique optimum. A fixed-momentum proximal-gradient
method uses its known strong-convexity constant, in span(G,V), without changing
the frozen unregularized fitting code used in the confirmatory experiment.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from . import operators
from .provenance import ROOT, digest, provenance, save_json


def fit_ridge(g, v, ridge, iterations=2000, tolerance=1e-9):
    # Orthogonal projection onto span(G,V) cannot increase the data residual
    # or Frobenius penalty, and preserves PSD. Extra SVD columns are harmless.
    basis = torch.linalg.svd(torch.cat((g,v),dim=-1),full_matrices=False)[0]
    reduced_g=basis.transpose(-1,-2)@g
    reduced_v=basis.transpose(-1,-2)@v
    gram=reduced_g@reduced_g.transpose(-1,-2)
    cross=reduced_v@reduced_g.transpose(-1,-2)
    penalty=torch.as_tensor(ridge,dtype=g.dtype,device=g.device)
    if penalty.ndim==0:
        penalty=penalty.expand(g.shape[0])
    if not bool((penalty>0).all()):
        raise ValueError("Strongly convex ridge fitting requires positive lambda")
    lipschitz=torch.linalg.eigvalsh(gram)[...,-1]+penalty
    step=(1/lipschitz)[:,None,None]
    beta=((lipschitz.sqrt()-penalty.sqrt())/(lipschitz.sqrt()+penalty.sqrt()))[:,None,None]
    k=torch.zeros_like(gram);y=k.clone()
    for iteration in range(1,iterations+1):
        gradient=operators.sym(y@gram-cross)+penalty[:,None,None]*y
        new=operators.project_psd(y-step*gradient)
        y=new+beta*(new-k)
        k=new
        if iteration%25==0 or iteration==iterations:
            gradient=operators.sym(k@gram-cross)+penalty[:,None,None]*k
            mapping=operators.norm(k-operators.project_psd(k-step*gradient))/operators.norm(k).clamp_min(1.)
            if bool((mapping<=tolerance).all()):
                break
    full=basis@k@basis.transpose(-1,-2)
    residual=operators.norm(full@g-v)
    return full,{"converged":mapping<=tolerance,"projected_gradient":mapping,
                 "iterations":iteration,"ridge":penalty,
                 "fit_relative_error":residual/operators.norm(v).clamp_min(1e-30),
                 "operator_frobenius_norm":operators.norm(full),
                 "objective":.5*residual.square()+.5*penalty*operators.norm(full).square()}


def analytic_checks():
    results=[]
    for ridge in (.002,.02,.2):
        # An anisotropic diagonal design tests iterative convergence against an
        # independent separable closed-form optimum, including a cone boundary.
        g=torch.diag(torch.tensor([1.,2.,3.,4.],dtype=torch.float64)).unsqueeze(0)
        v=torch.diag(torch.tensor([2.,-1.,1.,3.],dtype=torch.float64)).unsqueeze(0)
        expected=torch.diag(torch.tensor([2/(1+ridge),0.,3/(9+ridge),12/(16+ridge)],dtype=torch.float64)).unsqueeze(0)
        fitted,info=fit_ridge(g,v,ridge,tolerance=1e-12)
        torch.testing.assert_close(fitted,expected,rtol=1e-7,atol=1e-8)
        if not bool(info["converged"].all()):
            raise AssertionError("Analytic ridge case did not converge")
        results.append({"ridge":ridge,"max_absolute_error":(fitted-expected).abs().max(),
                        "iterations":info["iterations"],"projected_gradient":info["projected_gradient"]})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration",type=Path,default=Path("configs/inference/fit_sensitivity_v1.json"))
    parser.add_argument("--checkpoint",type=Path,default=Path("data/generated/inference/matched-development-p0.json"))
    parser.add_argument("--output",type=Path,default=Path("data/generated/inference/fit-sensitivity-development.json"))
    args=parser.parse_args()
    path=ROOT/args.configuration if not args.configuration.is_absolute() else args.configuration
    config=json.loads(path.read_text())
    parent=json.loads(args.checkpoint.read_text())
    if parent["split"]!="development" or parent["profile"]["name"]!="primary":
        raise ValueError("PSD fit sensitivity uses primary development networks only")
    if digest(parent["arrays"])!=parent["arrays_sha256"]:
        raise ValueError("Development checkpoint arrays failed verification")
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old["configuration_sha256"]==digest(path) and digest(old["arrays"])==old["arrays_sha256"]:
            print("Verified completed PSD fit sensitivity")
            return
        raise ValueError("Existing fit sensitivity conflicts with this configuration")
    torch.set_num_threads(1)
    metadata=provenance("cpu");started=time.perf_counter()
    checks=analytic_checks()
    data=torch.load(parent["arrays"],weights_only=True,map_location="cpu")
    count=2*parent["profile"]["outputs"]
    rows,models=[],[]
    for label in config["rules"]:
        observed=next(r for r in data["observations"] if r["checkpoint_step"]==config["checkpoint_step"] and r["rule"]==label and r["response_noise"]==0)
        g,v=observed["g"].transpose(-1,-2),observed["v"].transpose(-1,-2)
        gf,vf,gt,vt=g[...,:count],v[...,:count],g[...,count:],v[...,count:]
        for tolerance in config["pseudoinverse_tolerances"]:
            restriction,basis,info=operators.fit_isotropic_span(gf,vf,rtol=tolerance)
            prediction=restriction@(basis.transpose(-1,-2)@gt)
            error=operators.norm(prediction-vt)/operators.norm(vt).clamp_min(1e-30)
            rows.append({"rule":label,"method":"unregularized","rtol":tolerance,"ridge_fraction":0.,
                "prediction_error":torch.where(info["fit_attained"],error,torch.nan),
                "infimum_prediction_error":error,"operator_frobenius_norm":info["extension_frobenius_norm"],**info})
        spectrum=torch.linalg.eigvalsh(gf@gf.transpose(-1,-2))
        for fraction in config["ridge_fractions"]:
            penalty=fraction*spectrum[...,-1]
            fitted,info=fit_ridge(gf,vf,penalty,config["max_iterations"],config["stationarity_tolerance"])
            error=operators.norm(fitted@gt-vt)/operators.norm(vt).clamp_min(1e-30)
            rows.append({"rule":label,"method":"ridge","rtol":None,"ridge_fraction":fraction,
                "prediction_error":error,"fit_attained":torch.ones_like(info["converged"]),**info})
            models.append({"rule":label,"ridge_fraction":fraction,"operator":fitted})
            print(f"Ridge {label} fraction={fraction}: {int(info['converged'].sum())}/{len(parent['seeds'])} converged",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw_path=args.output.with_suffix(".pt");torch.save(models,raw_path)
    if metadata["sources"]!=provenance("cpu")["sources"]:
        raise RuntimeError("Sources changed during PSD sensitivity")
    save_json(args.output,{"configuration":config,"configuration_sha256":digest(path),
        "parent":str(args.checkpoint),"parent_sha256":digest(args.checkpoint),"seeds":parent["seeds"],
        "provenance":metadata,"seconds":time.perf_counter()-started,"analytic_checks":checks,
        "results":rows,"arrays":str(raw_path),"arrays_sha256":digest(raw_path)})


if __name__=="__main__":
    main()
