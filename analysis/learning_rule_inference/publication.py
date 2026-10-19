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

from .evaluate import write_csv
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


def tables_and_macros(rows, paired, mean, accuracy, predictions):
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
    tables_and_macros(rows, paired, mean, accuracy, predictions)
    names = ("fig_inference_primary", "fig_inference_robustness", "fig_inference_mean", "fig_inference_diagnostics")
    inputs = [DATA / name for name in ("inference_evaluation.csv", "inference_paired_comparison.csv", "inference_mean_evaluation.csv", "inference_mean_accuracy.csv", "inference_original_baseline.csv")]
    save_json(DATA / "inference_publication.provenance.json", {
        "script_sha256": digest(__file__), "inputs": {str(p):digest(p) for p in (*inputs, *extra_inputs)},
        "figure_hashes": {str(FIGURES/f"{n}.pdf"):digest(FIGURES/f"{n}.pdf") for n in names},
        "bootstrap": "2000 resamples of independent network summaries; fixed seeds",
        "source_evaluation_manifest": str(provenance_path), "source_evaluation_manifest_sha256": digest(provenance_path)})
    print("Built four publication figures and supporting tables")


if __name__ == "__main__":
    main()
