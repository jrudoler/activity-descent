"""Build paper figures and tables exclusively from verified completed experiments."""

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/activity-descent-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import wilson, write_csv
from .provenance import digest, save_json

DATA = Path("results/data")
FIGURES = Path("results/figures")
TABLES = Path("results/tables")
RULES = ("gd", "preconditioned", "feedback-0.1", "feedback-1", "node-1", "node-32", "node-512",
         "node-shared-1", "node-shared-512")
LABELS = ("GD", "PSD preconditioner", "Feedback 10%", "Feedback 100%", "Node R=1", "Node R=32", "Node R=512",
          "Shared node R=1", "Shared node R=512")
PROFILES = ("primary", "preconditioned-trained", "feedback-trained", "node-trained", "rank-two",
            "rank-sixteen", "width-thirty-two", "depth-three")
PROFILE_LABELS = ("Primary", "P-trained", "FA-trained", "NP-trained", "q=2", "q=16", "width 32", "depth 3")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#777777")


def read_csv(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def select(rows, **filters):
    found = [r for r in rows if all(str(r[k]) == str(v) for k, v in filters.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one completed result for {filters}, got {len(found)}")
    return found[0]


def finite(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    return a[np.isfinite(a)]


def median_ci(values):
    a = finite(values)
    if not len(a):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(91045)
    boot = np.median(a[rng.integers(len(a), size=(2000, len(a)))], axis=-1)
    return float(np.median(a)), *np.quantile(boot, [.025, .975]).tolist()


def errorbar(ax, positions, rows, color, label=None, horizontal=False, marker="o"):
    y = np.asarray([float(r["rejection_rate"]) for r in rows])
    lo, hi = (np.asarray([float(r[k]) for r in rows]) for k in ("ci_low", "ci_high"))
    error = np.maximum(0, np.stack((y - lo, hi - y)))
    if horizontal:
        ax.errorbar(y, positions, xerr=error, fmt=marker, ms=4, capsize=2, color=color, label=label)
    else:
        ax.errorbar(positions, y, yerr=error, fmt=marker+"-", ms=4, capsize=2, color=color, label=label)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight",
                    metadata={"Creator": "activity-descent inference analysis", "CreationDate": None,
                              "ModDate": None} if suffix == "pdf" else None)
    plt.close(fig)


def primary_figure(rows, paired):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1]}, layout="constrained")
    for i, (method, label) in enumerate((("operator", "Shared operator"), ("pearson", "Pearson"),
                                       ("cosine", "Cosine"), ("single_direction", "One direction"))):
        selected = [select(rows, profile="primary", checkpoint_step=1000, response_noise="0.0", method=method, rule=rule)
                    for rule in RULES]
        errorbar(axes[0], np.arange(len(RULES)) + (i-1.5)*.15, selected, COLORS[i], label, True)
    axes[0].set(yticks=range(len(RULES)), yticklabels=LABELS, xlim=(-.03, 1.04), xlabel="Held-out rejection rate",
                title="a  Same checkpoints, calibrated decisions")
    axes[0].invert_yaxis()
    axes[0].axvline(.05, color=".65", lw=1, ls=":")
    axes[0].legend(fontsize=8, loc="lower right")
    baselines = ("pearson", "cosine", "downhill", "population", "single_direction")
    for i, baseline in enumerate(baselines):
        r = select(paired, profile="primary", checkpoint_step=1000, response_noise="0.0", rule="feedback-0.1", baseline=baseline)
        center, low, high = (float(r[k]) for k in ("power_difference", "ci_low", "ci_high"))
        axes[1].errorbar(center, i, xerr=[[center-low], [high-center]], fmt="o", color=COLORS[0], capsize=3)
    axes[1].set(yticks=range(5), yticklabels=("Pearson", "Cosine", "Neuron signs", "Population descent", "One direction"),
                xlim=(-.03, 1.04), xlabel="Operator minus baseline detection rate", title="b  Feedback 10%: paired improvement")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color=".65", lw=1)
    save(fig, "fig_inference_primary")


def robustness_figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7), layout="constrained")
    for i, (method, label) in enumerate((("operator", "Shared operator"), ("pearson", "Pearson"), ("single_direction", "One direction"))):
        values = [select(rows, profile=p, checkpoint_step=1000, response_noise="0.0", method=method, rule="feedback-0.1") for p in PROFILES]
        errorbar(axes[0, 0], np.arange(8)+(i-1)*.15, values, COLORS[i], label)
    axes[0, 0].set(xticks=range(8), xticklabels=PROFILE_LABELS, title="a  Training and architecture controls", ylabel="Feedback 10% detection")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].legend(fontsize=8)
    for i, p in enumerate(PROFILES[:4]):
        selected = [select(rows, profile=p, checkpoint_step=t, response_noise="0.0", method="operator", rule="feedback-0.1") for t in (0, 200, 1000)]
        errorbar(axes[0, 1], [0, 200, 1000], selected, COLORS[i], PROFILE_LABELS[i])
    axes[0, 1].set(xlabel="Training updates", ylabel="Feedback 10% detection", title="b  Equal-step checkpoints")
    axes[0, 1].legend(fontsize=8)
    for i, noise in enumerate(("0.0", "0.01")):
        selected = [select(rows, profile=p, checkpoint_step=1000, response_noise=noise, method="operator", rule="feedback-0.1") for p in PROFILES]
        errorbar(axes[1, 0], np.arange(8)+(i-.5)*.16, selected, COLORS[i], "No noise" if i==0 else "1% GD-RMS noise")
        selected = [select(rows, profile="primary", checkpoint_step=1000, response_noise=noise, method="operator", rule=r)
                    for r in ("gd", "preconditioned", "node-shared-1", "node-shared-512")]
        errorbar(axes[1, 1], np.arange(4)+(i-.5)*.16, selected, COLORS[i], "No noise" if i==0 else "1% GD-RMS noise")
    axes[1, 0].set(xticks=range(8), xticklabels=PROFILE_LABELS, ylabel="Feedback 10% detection", title="c  Observation noise limits discrimination")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(xticks=range(4), xticklabels=("GD", "PSD-P", "Shared R=1", "Shared R=512"), ylabel="Rejection rate", title="d  Calibration does not cover every PSD rule")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].axhline(.05, color=".6", ls=":")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_ylim(-.04, 1.04)
    save(fig, "fig_inference_robustness")


def mean_figure(rows, accuracy):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), layout="constrained")
    counts = sorted({int(r["samples"]) for r in rows})
    for i, rule in enumerate(("directional-null", "preconditioned-null", "node")):
        for ax, method in ((axes[0, 0], "uncertainty_aware"), (axes[0, 1], "naive_deterministic_cutoff")):
            values = [select(rows, mode="responses-first", samples=n, rule=rule, method=method) for n in counts]
            errorbar(ax, counts, values, COLORS[i], ("Directional GD", "Directional PSD-P", "Physical node")[i])
    for i, rule in enumerate(("feedback-0.1", "feedback-1")):
        for j, mode in enumerate(("responses-first", "directions-first")):
            values = [select(rows, mode=mode, samples=n, rule=rule, method="uncertainty_aware") for n in counts]
            errorbar(axes[1, 0], counts, values, COLORS[i], f"Feedback {('10%', '100%')[i]}, {('responses', 'directions')[j]}", marker=("o", "s")[j])
    for i, rule in enumerate(("directional-null", "preconditioned-null", "node")):
        values = [select(accuracy, mode="directions-first", samples=n, rule=rule, metric="relative_error_to_deterministic_mean_step") for n in counts]
        median, lo, hi = (np.asarray([float(r[k]) for r in values]) for k in ("median", "ci_low", "ci_high"))
        axes[1, 1].plot(counts, median, "o-", ms=4, color=COLORS[i], label=("Directional GD", "Directional PSD-P", "Physical node")[i])
        axes[1, 1].fill_between(counts, lo, hi, color=COLORS[i], alpha=.12)
    axes[0, 0].set(title="a  Monte Carlo uncertainty included", ylabel="Rejection rate")
    axes[0, 1].set(title="b  Naive deterministic cutoff", ylabel="Rejection rate")
    axes[1, 0].set(title="c  Power against biased conditional means", ylabel="Rejection rate")
    axes[1, 1].set(title="d  Convergence to mean-direction reference", ylabel="Median relative response error", yscale="log")
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Independent perturbation draws R")
        ax.legend(fontsize=7, loc="best")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_ylim(-.04, 1.04)
        ax.axhline(.05, color=".6", ls=":", lw=1)
    save(fig, "fig_inference_mean")


def diagnostics_figure():
    baseline = read_csv("inference_original_baseline.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), layout="constrained")
    summaries = []
    for i, metric in enumerate(("pearson", "downhill_fraction", "full_relative_error", "diagonal_relative_error")):
        medians, lows, highs = [], [], []
        for layer in range(4):
            by_seed = [[float(r[metric]) for r in baseline if int(r["seed"])==seed and int(r["layer"])==layer and r[metric]] for seed in range(20)]
            values = [np.nanmedian(v) for v in by_seed]
            center, low, high = median_ci(values)
            medians.append(center); lows.append(low); highs.append(high)
            summaries.append({"layer": layer, "metric": metric, "independent_runs": len(finite(values)), "median": center, "ci_low": low, "ci_high": high})
        ax = axes[0, 0] if i<2 else axes[0, 1]
        ax.errorbar(np.arange(4)+(i%2-.5)*.08, medians, yerr=[np.asarray(medians)-lows, np.asarray(highs)-medians],
                    fmt="o-", ms=4, capsize=2, color=COLORS[i%2], label=("Pearson", "Downhill fraction", "Full kernel", "Diagonal kernel")[i])
    for ax in axes[0]:
        ax.set_xticks(range(4), ("Hidden 1", "Hidden 2", "Hidden 3", "Output"))
        ax.legend(fontsize=8)
    axes[0, 0].set(title="a  Original classification task, layerwise", ylabel="Median over time, then over networks")
    axes[0, 1].set(title="b  First-order fit versus diagonal approximation", ylabel="Relative prediction error", yscale="log")
    pilot_path = Path("data/generated/inference/pilot-q8-trained.json")
    pilot = json.loads(pilot_path.read_text())
    if digest(pilot["checkpoint_path"]) != pilot["checkpoint_sha256"]:
        raise ValueError("Development mechanism arrays failed verification")
    for i, mask in enumerate(("all", "own", "upstream")):
        records = [r for r in pilot["mechanisms"] if r["parameter_mask"] == mask]
        etas, full, diagonal = [], [], []
        for r in sorted(records, key=lambda r:r["eta"]):
            layer = r["layers"][pilot["configuration"]["depth"]-1]
            etas.append(r["eta"])
            full.append(float(np.nanmedian(layer["full_relative_error"])))
            diagonal.append(float(np.nanmedian(layer["diagonal_relative_error"])))
        axes[1, 0].loglog(etas, full, "o-", ms=4, color=COLORS[i], label=f"{mask}: full")
        axes[1, 0].loglog(etas, diagonal, "s--", ms=3, color=COLORS[i], label=f"{mask}: diagonal")
    axes[1, 0].set(title="c  Parameter freezing and step size (development)", xlabel="Step size", ylabel="Median relative prediction error")
    axes[1, 0].legend(fontsize=7, ncol=2)
    sensitivity_path = Path("data/generated/inference/sensitivity-development.json")
    sensitivity = json.loads(sensitivity_path.read_text())
    if digest(sensitivity["arrays"]) != sensitivity["arrays_sha256"]:
        raise ValueError("Sensitivity raw arrays failed verification")
    for i, (variant, sigma, label) in enumerate((("exact-shared-PSD", None, "Exact shared PSD"),
                                                ("physical-node", 1e-4, "Node sigma=1e-4"), ("physical-node", 1e-6, "Node sigma=1e-6"))):
        records = sorted([r for r in sensitivity["results"] if r["variant"] == variant and r["sigma"] == sigma and r["repeats"]==1], key=lambda r:r["eta"])
        axes[1, 1].loglog([r["eta"] for r in records], [np.nanmedian(r["operator"]) for r in records], "o-", ms=4, color=COLORS[i], label=label)
        axes[1, 1].axhline(np.nanmedian(records[0]["first_order_operator"]), color=COLORS[i], ls=":", lw=1)
    axes[1, 1].set(title="d  Shared directions: finite-step residual", xlabel="Step size", ylabel="Median operator score")
    axes[1, 1].legend(fontsize=8)
    save(fig, "fig_inference_diagnostics")
    write_csv(DATA / "inference_layerwise_summary.csv", summaries)
    return [pilot_path, sensitivity_path]


def supporting_tables(manifests):
    grouped = {}
    for manifest in manifests:
        path = Path(manifest["path"])
        if digest(path) != manifest["sha256"]:
            raise ValueError(f"Main result changed since verification: {path}")
        d = json.loads(path.read_text())
        if d["split"] != "evaluation":
            continue
        for row in d["results"]:
            if row["checkpoint_step"]!=1000 or row["response_noise"]!=0:
                continue
            key = d["profile"]["name"], row["rule"]
            store = grouped.setdefault(key, {})
            for name, values in row["prediction"].items():
                if isinstance(values, list) and values and isinstance(values[0], (int, float, bool, type(None))):
                    store.setdefault(name, []).extend(values)
            for name in ("operator",):
                store.setdefault(name, []).extend(row["scores"][name])
            store.setdefault("loss", []).extend(row["loss"])
    summary = []
    for (profile, rule), values in grouped.items():
        for metric, a in values.items():
            if len(a)!=200:
                raise ValueError(f"Incomplete supporting statistic: {profile}, {rule}, {metric}")
            good = finite(a)
            summary.append({"profile": profile, "rule": rule, "metric": metric, "evaluation_runs": len(good),
                "undefined_runs": len(a)-len(good), "median": np.median(good) if len(good) else math.nan,
                "q25": np.quantile(good,.25) if len(good) else math.nan,
                "q75": np.quantile(good,.75) if len(good) else math.nan,
                "true_count": int(np.sum(good)) if metric in ("fit_attained", "prediction_supported", "converged") else ""})
    write_csv(DATA / "inference_prediction_summary.csv", summary)
    return summary


def development_controls():
    paths = [Path("data/generated/inference/mechanism-controls-development.json"),
             Path("data/generated/inference/fit-sensitivity-development.json")]
    controls, fitting = (json.loads(p.read_text()) for p in paths)
    for report in (controls, fitting):
        if digest(report["arrays"]) != report["arrays_sha256"] or digest(report["parent"]) != report["parent_sha256"]:
            raise ValueError("Development control arrays or parent reports changed")
        if len(report["seeds"]) != 20:
            raise ValueError("Expected 20 independent development networks")
    metrics = ("full_relative_error", "diagonal_relative_error", "cosine", "pearson", "downhill_fraction",
               "population_descent_rate", "self_term_l2", "interference_l2", "gate_on", "gate_off",
               "gradient_dot_activity_change", "actual_loss_change")
    layer_summary = []
    for row in controls["layer_loss_results"]:
        for metric in metrics:
            values = np.asarray(row[metric], dtype=float).reshape(-1)
            center, low, high = median_ci(values)
            layer_summary.append({**{k:row[k] for k in ("loss_control", "parameter_mask", "eta", "layer", "layer_type")},
                "metric":metric, "networks":len(finite(values)), "undefined_networks":len(values)-len(finite(values)),
                "median":center, "ci_low":low, "ci_high":high})
    write_csv(DATA / "inference_mechanism_controls.csv", layer_summary)
    scale_summary = []
    for row in controls["scale_results"]:
        center, low, high = median_ci(row["operator"])
        scale_summary.append({"rule":row["rule"], "mode":row["mode"], "networks":len(finite(row["operator"])),
            "median":center, "ci_low":low, "ci_high":high,
            "q25":float(np.quantile(row["operator"],.25)), "q75":float(np.quantile(row["operator"],.75)),
            "median_scale":float(np.median(row["magnitude_scale"])),
            "max_first_order_scale_invariance_error":max(row["first_order_scale_invariance_error"]),
            "median_relative_first_order_error":float(np.median(row["relative_first_order_error"]))})
    write_csv(DATA / "inference_scale_sensitivity.csv", scale_summary)
    fig, axes = plt.subplots(2, 2, figsize=(10.2,7), layout="constrained")
    for i, loss in enumerate(("task", "synthetic-layer")):
        for j, metric in enumerate(("full_relative_error", "diagonal_relative_error")):
            selected = sorted([r for r in layer_summary if r["loss_control"]==loss and r["parameter_mask"]=="all"
                               and r["layer"]==1 and r["metric"]==metric], key=lambda r:r["eta"])
            x=[r["eta"] for r in selected];y=[r["median"] for r in selected]
            axes[0,0].loglog(x,y,("o-","s--")[j],ms=4,color=COLORS[i],label=f"{loss}: {('full','diagonal')[j]}")
            axes[0,0].fill_between(x,[r["ci_low"] for r in selected],[r["ci_high"] for r in selected],color=COLORS[i],alpha=.1)
        centers,lows,highs=[],[],[]
        for mask in ("all","own","upstream"):
            row=next(r for r in controls["layer_loss_results"] if r["loss_control"]==loss and r["parameter_mask"]==mask and r["eta"]==1e-5 and r["layer"]==1)
            numerator=np.asarray(row["interference_l2"],dtype=float).reshape(-1)
            denominator=np.asarray(row["self_term_l2"],dtype=float).reshape(-1)
            values=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>1e-20)
            center,low,high=median_ci(values);centers.append(center);lows.append(low);highs.append(high)
        axes[0,1].errorbar(np.arange(3)+(i-.5)*.12,centers,
            yerr=[np.asarray(centers)-lows,np.asarray(highs)-centers],fmt="o-",ms=4,capsize=2,color=COLORS[i],label=loss)
        selected=sorted([r for r in controls["layer_loss_results"] if r["loss_control"]==loss and r["parameter_mask"]=="all" and r["layer"]==1],key=lambda r:r["eta"])
        rates,lows,highs=[],[],[]
        for row in selected:
            counts=np.asarray(row["gate_on"])+np.asarray(row["gate_off"])
            total=len(counts);changed=int((counts.reshape(-1)>0).sum());low,high=wilson(changed,total)
            rates.append(changed/total);lows.append(low);highs.append(high)
        axes[1,0].errorbar([r["eta"] for r in selected],rates,
            yerr=[np.asarray(rates)-lows,np.asarray(highs)-rates],fmt="o-",ms=4,capsize=2,color=COLORS[i],label=loss)
    for i,mode in enumerate(("raw","matched")):
        selected=[select(scale_summary,rule=rule,mode=mode) for rule in RULES]
        centers=np.asarray([r["median"] for r in selected]);low=np.asarray([r["ci_low"] for r in selected]);high=np.asarray([r["ci_high"] for r in selected])
        axes[1,1].errorbar(centers,np.arange(9)+(i-.5)*.15,xerr=[centers-low,high-centers],fmt="o",ms=3,capsize=2,color=COLORS[i],label=mode)
    axes[0,0].set(title="a  Task and complete-layer loss gradients",xlabel="Step size",ylabel="Median relative prediction error")
    axes[0,1].set(title="b  Parameter freezing isolates interference",xticks=range(3),xticklabels=("All","Own layer","Upstream"),ylabel="Median ||interference|| / ||self||")
    axes[1,0].set(title="c  Observed-layer gate changes",xlabel="Step size",ylabel="Fraction of development networks",xscale="log",ylim=(-.03,1.03))
    axes[1,1].set(title="d  Same draws, constant bank rescaling",xlabel="Median compatibility score",xscale="log",yticks=range(9),yticklabels=LABELS)
    axes[1,1].invert_yaxis();axes[1,1].tick_params(axis="y",labelsize=7)
    for ax in axes.flat:ax.legend(fontsize=7)
    save(fig,"fig_inference_mechanism_controls")
    fit_summary=[]
    for row in fitting["results"]:
        converged=np.asarray(row["converged"],dtype=bool)
        attained=np.asarray(row["fit_attained"],dtype=bool)
        predicted=np.asarray(row["prediction_error"],dtype=float)
        predicted=np.where(converged & attained,predicted,np.nan)
        center,low,high=median_ci(predicted)
        norms=np.asarray(row["operator_frobenius_norm"],dtype=float)
        norms=np.where(converged & attained,norms,np.nan)
        fit_summary.append({"rule":row["rule"],"method":row["method"],"rtol":row["rtol"],"ridge_fraction":row["ridge_fraction"],
            "networks":len(attained),"converged":int(converged.sum()),"valid_psd_fits":int((converged&attained).sum()),
            "prediction_networks":len(finite(predicted)),"prediction_median":center,"prediction_ci_low":low,"prediction_ci_high":high,
            "operator_norm_median":float(np.median(finite(norms))) if len(finite(norms)) else math.nan,
            "fit_error_median":float(np.median(row["fit_relative_error"])),
            "max_projected_gradient":float(np.max(row["projected_gradient"])),"iterations":row["iterations"]})
    write_csv(DATA / "inference_fit_sensitivity.csv",fit_summary)
    def number(value):
        if not math.isfinite(value):return "---"
        mantissa,exponent=f"{value:.2e}".split("e")
        return f"${mantissa}\\cdot10^{{{int(exponent)}}}$"
    table=[r"\begin{tabular}{lrrrr}",r"\hline",r"Rule & $\lambda/c$ & PSD fits & Prediction error & $\|K\|_F$\\",r"\hline"]
    for rule in fitting["configuration"]["rules"]:
        selected=[r for r in fit_summary if r["rule"]==rule and (r["method"]=="ridge" or r["rtol"]==1e-9)]
        for row in selected:
            label=LABELS[RULES.index(rule)].replace("%",r"\%")
            table.append(f"{label} & {row['ridge_fraction']:g} & {row['valid_psd_fits']}/20 & " +
                         number(row["prediction_median"]) + " & " + number(row["operator_norm_median"]) + r"\\")
    table.extend([r"\hline",r"\end{tabular}"])
    TABLES.mkdir(parents=True,exist_ok=True)
    (TABLES / "inference_regularization.tex").write_text("\n".join(table)+"\n")
    return paths


def gpu_replication_table(cpu_rows):
    paths = [DATA / name for name in ("inference_gpu_evaluation.csv", "inference_gpu_evaluation.provenance.json",
             "inference_backend_validation.csv", "inference_backend_validation.provenance.json",
             "inference_cuda_replication_validation.json")]
    gpu, backend = read_csv(paths[0].name), read_csv(paths[2].name)
    manifest, comparison, validation = [json.loads(paths[i].read_text()) for i in (1, 3, 4)]
    cpu_hash = digest(DATA / "inference_evaluation.csv")
    science = json.loads((DATA / "inference_evaluation.provenance.json").read_text())["science_sources"]
    if (manifest["cpu_evaluation_sha256"] != cpu_hash or comparison["cpu_evaluation_sha256"] != cpu_hash
            or manifest["science_sources"] != science
            or manifest["configuration_sha256"] != digest("configs/inference/confirmatory_v1.json")
            or manifest["evaluator_sha256"] != digest(Path(__file__).with_name("evaluate_gpu.py"))):
        raise ValueError("CUDA evidence does not match the frozen CPU study or evaluator")
    if len(gpu) != 432 or len(backend) != 12 or len(manifest["completed_shards"]) != 10:
        raise ValueError("Incomplete CUDA replication or saved-checkpoint comparison")
    for metadata in (comparison["provenance"], validation["provenance"]):
        if metadata["device"] != "cuda" or "B200" not in metadata.get("gpu", ""):
            raise ValueError("Actual B200 execution is required for the CUDA evidence")
        if any(metadata["sources"][f"analysis/learning_rule_inference/{name}"] != wanted
               for name, wanted in science.items()):
            raise ValueError("Saved-checkpoint comparison or validation used changed science sources")
    if not validation["passed"] or validation["tests"] != 18 or validation["failures"]:
        raise ValueError("The CUDA replication scientific checks did not pass")
    for record in (*manifest["completed_shards"], *comparison["parents"]):
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"CUDA evidence parent changed: {path}")
        paths.append(path)
    if any(int(r["networks"]) != 200 or float(r["max_relative_response_error"]) > 1e-7 for r in backend):
        raise ValueError("Saved-checkpoint CUDA comparison is incomplete or outside its tolerance")
    table = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"& \multicolumn{2}{c}{Shared operator} & \multicolumn{2}{c}{Pearson}\\",
             r"Rule & CPU & CUDA & CPU & CUDA\\", r"\hline"]
    for rule, label in zip(RULES, LABELS):
        cells = []
        for method in ("operator", "pearson"):
            reference = select(cpu_rows, profile="primary", checkpoint_step=1000,
                               response_noise="0.0", rule=rule, method=method)
            result = select(gpu, checkpoint_step=1000, response_noise="0.0", rule=rule, method=method)
            if (int(result["evaluation_runs"]) != 200 or int(result["undefined_runs"]) != 0
                    or result["cpu_rejections"] != reference["rejections"]
                    or float(result["cpu_cutoff"]) != float(reference["cutoff"])):
                raise ValueError("CUDA table requires complete evaluation with the original CPU cutoffs")
            cells.extend((reference["rejections"], result["rejections"]))
        table.append(label.replace("%", r"\%") + " & " + " & ".join(cells) + r"\\")
    table += [r"\hline", r"\end{tabular}"]
    (TABLES / "inference_gpu_replication.tex").write_text("\n".join(table)+"\n")
    macros = {name:select(gpu, checkpoint_step=1000, response_noise="0.0", rule=rule, method=method)["rejections"]
              for name, rule, method in (("GPUFeedbackRejections", "feedback-0.1", "operator"),
                  ("GPUPearsonRejections", "feedback-0.1", "pearson"), ("GPUGDRejections", "gd", "operator"),
                  ("GPUPreconditionedRejections", "preconditioned", "operator"))}
    macros["GPUDecisionDisagreements"] = str(sum(int(r["decision_disagreements"]) for r in backend))
    max_error = max(float(r["max_relative_response_error"]) for r in backend)
    mantissa, exponent = f"{max_error:.2e}".split("e")
    macros["GPUResponseMaxError"] = f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return paths, macros


def tables_and_macros(rows, paired, mean, accuracy, predictions, gpu_macros):
    TABLES.mkdir(parents=True, exist_ok=True)
    table = [r"\begin{tabular}{lrrr}", r"\hline", r"Update rule & Operator & Pearson & One direction\\", r"\hline"]
    for rule, label in zip(RULES, LABELS):
        values = [select(rows, profile="primary", checkpoint_step=1000, response_noise="0.0", method=m, rule=rule)
                  for m in ("operator", "pearson", "single_direction")]
        cells = [f"{r['rejections']}/{r['evaluation_runs']}" for r in values]
        table.append(label.replace("%", r"\%") + " & " + " & ".join(cells) + r"\\")
    table += [r"\hline", r"\end{tabular}"]
    (TABLES / "inference_primary.tex").write_text("\n".join(table)+"\n")
    def number(value):
        value = float(value)
        if not math.isfinite(value):
            return "---"
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\cdot10^{{{int(exponent)}}}$"
    table = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Rule & Attained & PSD error & Linear error & Diagonal error\\", r"\hline"]
    for rule, label in zip(RULES, LABELS):
        values = {metric:select(predictions, profile="primary", rule=rule, metric=metric)
                  for metric in ("fit_attained", "psd_prediction_error", "linear_prediction_error", "diagonal_prediction_error")}
        cells = [f"{values['fit_attained']['true_count']}/200"]
        cells += [number(values[metric]["median"]) for metric in ("psd_prediction_error", "linear_prediction_error", "diagonal_prediction_error")]
        table.append(label.replace("%", r"\%") + " & " + " & ".join(cells) + r"\\")
    table += [r"\hline", r"\end{tabular}"]
    (TABLES / "inference_prediction.tex").write_text("\n".join(table)+"\n")
    pair = select(paired, profile="primary", checkpoint_step=1000, response_noise="0.0", rule="feedback-0.1", baseline="pearson")
    macros = {"PrimaryPowerDifference": f"{100*float(pair['power_difference']):.1f}",
              "PrimaryPowerDifferenceLow": f"{100*float(pair['ci_low']):.1f}",
              "PrimaryPowerDifferenceHigh": f"{100*float(pair['ci_high']):.1f}"}
    for rule, name in (("directional-null", "Directional"), ("preconditioned-null", "MeanPreconditioned"), ("node", "PhysicalNode"), ("feedback-0.1", "MeanFeedback")):
        for count in (8,8192):
            r = select(mean, mode="responses-first", samples=count, rule=rule, method="uncertainty_aware")
            macros[f"{name}{'Small' if count==8 else 'Large'}Rejections"] = r["rejections"]
    for rule, name in (("directional-null", "Directional"), ("node", "PhysicalNode")):
        r = select(accuracy, mode="directions-first", samples=8192, rule=rule, metric="relative_error_to_deterministic_mean_step")
        macros[f"{name}LargeError"] = f"{100*float(r['median']):.2f}"
    macros.update(gpu_macros)
    (TABLES / "inference_numbers.tex").write_text("% Generated from verified completed experiments.\n" +
        "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key,value in macros.items()))


def main():
    plt.rcParams.update({"font.size":9, "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42})
    rows, paired = read_csv("inference_evaluation.csv"), read_csv("inference_paired_comparison.csv")
    mean, accuracy = read_csv("inference_mean_evaluation.csv"), read_csv("inference_mean_accuracy.csv")
    provenance_path = DATA / "inference_evaluation.provenance.json"
    manifests = json.loads(provenance_path.read_text())["completed_shards"]
    predictions = supporting_tables(manifests)
    primary_figure(rows, paired)
    robustness_figure(rows)
    mean_figure(mean, accuracy)
    extra_inputs = diagnostics_figure()
    extra_inputs.extend(development_controls())
    gpu_inputs, gpu_macros = gpu_replication_table(rows)
    extra_inputs.extend(gpu_inputs)
    tables_and_macros(rows, paired, mean, accuracy, predictions, gpu_macros)
    names = ("fig_inference_primary", "fig_inference_robustness", "fig_inference_mean", "fig_inference_diagnostics", "fig_inference_mechanism_controls")
    inputs = [DATA / name for name in ("inference_evaluation.csv", "inference_paired_comparison.csv", "inference_mean_evaluation.csv", "inference_mean_accuracy.csv", "inference_original_baseline.csv")]
    save_json(DATA / "inference_publication.provenance.json", {
        "script_sha256": digest(__file__), "inputs": {str(p):digest(p) for p in (*inputs, *extra_inputs)},
        "figure_hashes": {str(FIGURES/f"{n}.pdf"):digest(FIGURES/f"{n}.pdf") for n in names},
        "bootstrap": "2000 resamples of independent network summaries; fixed seeds",
        "source_evaluation_manifest": str(provenance_path), "source_evaluation_manifest_sha256": digest(provenance_path)})
    print(f"Built {len(names)} publication figures and supporting tables")


if __name__ == "__main__":
    main()
