"""Summarize completed development pilots without treating probes as replicates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(paths):
    rows, inputs = [], []
    for path in paths:
        data = json.loads(path.read_text())
        configuration = data["configuration"]
        if configuration["mode"] != "pilot":
            raise ValueError(f"Not a completed pilot: {path}")
        inputs.append({"path": str(path), "provenance": data["provenance"],
                       "checkpoint_sha256": data["checkpoint_sha256"]})
        for result in data["results"]:
            row = {"input": str(path), "outputs": configuration["outputs"],
                   "width": configuration["width"], "depth": configuration["depth"],
                   "train_steps": configuration["train_steps"], "networks": len(data["seeds"]),
                   "rule": result["rule"], "eta": result["eta"],
                   "node_repeats": result["node_repeats"], "shared_noise": result["shared_noise"]}
            for name in ("reciprocity", "negative_quadratic", "linearity", "nullspace", "gradient_rank"):
                values = np.asarray(result["scores"][name], dtype=float)
                row[f"{name}_median"] = float(np.median(values))
                row[f"{name}_q10"] = float(np.quantile(values, .1))
                row[f"{name}_q90"] = float(np.quantile(values, .9))
            for name in ("cosine", "pearson", "downhill_fraction"):
                # First reduce the repeated probes within each independent run.
                values = np.asarray(result["alignment"][name], dtype=float)
                within_run = np.nanmedian(values, axis=-1)
                row[f"{name}_median"] = float(np.nanmedian(within_run))
            for name in ("psd_prediction_error", "linear_prediction_error", "diagonal_prediction_error",
                         "fit_relative_error", "projected_gradient"):
                row[f"{name}_median"] = float(np.nanmedian(np.asarray(result["prediction"][name], dtype=float)))
            row["fit_converged_fraction"] = float(np.mean(result["prediction"]["converged"]))
            rows.append(row)
    return rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/data/inference_pilot_summary.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/figures/fig_inference_pilot.pdf"))
    args = parser.parse_args()
    rows, sources = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".provenance.json").write_text(json.dumps({"sources": sources,
        "interpretation": "Development pilots; quantiles summarize independent networks and are not confidence intervals."}, indent=2) + "\n")
    # A development figure, with independent-network quantile ranges.
    panels = sorted(set((r["outputs"], r["train_steps"]) for r in rows))
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), squeeze=False)
    colors = {"gd": "#157f6d", "preconditioned": "#4d74b3", "feedback": "#c35546",
              "node": "#8856a7"}
    for ax, (outputs, steps) in zip(axes[0], panels):
        for row in rows:
            if (row["outputs"], row["train_steps"]) != (outputs, steps):
                continue
            marker = "s" if row["shared_noise"] else "o"
            alpha = .45 if row["rule"] == "node" else .9
            ax.scatter(row["cosine_median"], max(row["reciprocity_median"], 1e-12),
                       c=colors[row["rule"]], marker=marker, alpha=alpha, s=25)
        ax.set_yscale("log")
        ax.set_xlabel("Median within-run cosine alignment")
        ax.set_ylabel("Reciprocity residual")
        ax.set_title(f"{outputs} outputs; {steps} training steps")
        ax.grid(alpha=.15)
    figure.suptitle("Development pilot: alignment and joint-probe consistency", fontsize=11)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=name)
               for name, c in colors.items()]
    handles.append(plt.Line2D([], [], marker="s", linestyle="", color="#666666",
                              label="shared perturbations"))
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, .1, 1, .94))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, bbox_inches="tight")
    figure.savefig(args.figure.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"Saved {len(rows)} condition summaries to {args.output}")


if __name__ == "__main__":
    main()
