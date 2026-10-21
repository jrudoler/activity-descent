"""Reproduce learning-rule inference; retain the original figure workflows."""

import json

INFERENCE_CONFIG = json.load(open("configs/inference/confirmatory_v1.json"))
INFERENCE_PROFILES = [p["name"] for p in INFERENCE_CONFIG["profiles"]]
INFERENCE_SOURCES = [f"analysis/learning_rule_inference/{p}.py"
                     for p in ("core", "operators", "campaign", "diagnostics", "provenance")]
INFERENCE_REPORTS = [f"data/generated/inference/confirmatory-v1/{profile}-{split}-s{shard:02}.json"
                    for profile in INFERENCE_PROFILES for split in ("calibration", "evaluation")
                    for shard in range(10)]
MEAN_REPORTS = [f"data/generated/inference/mean-v1/primary-{split}-s{shard:02}.json"
                for split in ("calibration", "evaluation") for shard in range(10)]
INFERENCE_FIGURE_NAMES = ["fig_inference_primary", "fig_inference_robustness",
                          "fig_inference_mean", "fig_inference_diagnostics", "fig_inference_mechanism_controls"]
INFERENCE_FIGURES = [f"results/figures/{n}.{s}" for n in INFERENCE_FIGURE_NAMES for s in ("pdf", "png")]
INFERENCE_TABLES = [f"results/tables/inference_{n}.tex" for n in ("primary", "numbers", "prediction", "regularization", "gpu_replication")]
INFERENCE_STAGING = "paper/research/learning_rule_inference/generated"
INFERENCE_PAPER_FIGURES = [f"{INFERENCE_STAGING}/figures/{n}.pdf" for n in INFERENCE_FIGURE_NAMES]
INFERENCE_PAPER_TABLES = [f"{INFERENCE_STAGING}/inference_{n}.tex" for n in ("primary", "numbers", "prediction", "regularization", "gpu_replication", "hardware", "hardware_b200")]
HARDWARE_CONFIG = json.load(open("configs/inference/hardware_mig45_v1.json"))
HARDWARE_INPUTS = [f"{HARDWARE_CONFIG['input_directory']}/{name}"
                  for device in ("cpu", "cuda")
                  for job in [HARDWARE_CONFIG['cpu_job_id' if device == 'cpu' else 'gpu_job_id']]
                  for name in ([f"validation-{device}-{job}.json"] +
                               [f"benchmark-{device}-n{n}-{job}.json" for n in HARDWARE_CONFIG['batch_sizes']])]
HARDWARE_INPUTS.append(f"{HARDWARE_CONFIG['input_directory']}/{HARDWARE_CONFIG['accounting']}")
HARDWARE_B200_CONFIG = json.load(open("configs/inference/hardware_b200_v1.json"))
HARDWARE_INPUTS += [f"{HARDWARE_B200_CONFIG['input_directory']}/{name}" for name in
                   ["validation-cuda-9156748.json", "benchmark-cuda-n1-9156748.json",
                    "benchmark-cuda-n20-9156748.json", "benchmark-cuda-n64-9156748.json",
                    HARDWARE_B200_CONFIG['accounting']]]

MAIN_FIGURE_NAMES = [
    "fig_main_compact",
    "fig_width_sweep",
    "fig_depth_sweep",
    "fig_width_sweep_appendix",
]
MAIN_FIGURES = [
    f"results/figures/{name}.{suffix}"
    for name in MAIN_FIGURE_NAMES
    for suffix in ("pdf", "png")
]
PAPER_FIGURES = [
    f"paper/generated/figures/{name}.pdf" for name in MAIN_FIGURE_NAMES
]
LEGACY_RESEARCH_FIGURES = [f"paper/research/legacy_figures/{name}.pdf" for name in MAIN_FIGURE_NAMES]
DROPOUT_FIGURES = [
    f"results/figures/{name}.{suffix}"
    for name in ("fig_dropout_sweep", "fig_dropout_phi_heatmaps")
    for suffix in ("pdf", "png")
]
rule all:
    input:
        "paper/activity_dynamics.pdf",


rule inference_research:
    input:
        INFERENCE_FIGURES,
        INFERENCE_PAPER_FIGURES,
        INFERENCE_PAPER_TABLES,


rule legacy_figures:
    input:
        MAIN_FIGURES,
        DROPOUT_FIGURES,
        "data/generated/dropout_results.json",
        "results/data/dropout_summary.csv",
        LEGACY_RESEARCH_FIGURES,


rule generate_main_figures:
    input:
        script="analysis/generate_main_figures/run.py",
        lockfile="uv.lock",
    output:
        MAIN_FIGURES,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output-dir results/figures"


rule run_dropout:
    input:
        script="analysis/run_dropout/run.py",
        lockfile="uv.lock",
    output:
        data="data/generated/dropout_results.json",
        summary="results/data/dropout_summary.csv",
        figures=DROPOUT_FIGURES,
    shell:
        """
        PYTHONPATH=src uv run python {input.script} \
          --data-output {output.data} \
          --summary-output {output.summary} \
          --figure-output-dir results/figures
        """


rule collect_paper_assets:
    input:
        figures=[f"results/figures/{name}.pdf" for name in MAIN_FIGURE_NAMES],
    output:
        figures=LEGACY_RESEARCH_FIGURES,
    shell:
        """
        mkdir -p paper/research/legacy_figures
        cp {input.figures} paper/research/legacy_figures/
        """


rule compile_paper:
    input:
        manuscript="paper/activity_dynamics.tex",
        references="paper/activity_dynamics.bib",
        latexmk_config="paper/latexmkrc",
        figures=PAPER_FIGURES + ["paper/generated/figures/fig_ntk.pdf"],
        baseline="paper/manuscript_baseline.json",
        builder="tools/build_paper.py",
        template=["paper/jmlr.cls", "paper/jmlrutils.sty"],
    output:
        "paper/activity_dynamics.pdf",
    shell:
        "python tools/build_paper.py"


rule inference_shard:
    input:
        configuration="configs/inference/confirmatory_v1.json",
        sources=INFERENCE_SOURCES,
        lockfile="uv.lock",
    output:
        report="data/generated/inference/confirmatory-v1/{profile}-{split}-s{shard}.json",
        arrays="data/generated/inference/confirmatory-v1/{profile}-{split}-s{shard}.pt",
    wildcard_constraints:
        profile="|".join(INFERENCE_PROFILES),
        split="calibration|evaluation",
        shard="[0-9]{2}",
    params:
        index=lambda w: INFERENCE_PROFILES.index(w.profile),
        shard=lambda w: int(w.shard),
    threads: 1
    resources:
        mem_mb=5632,
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.campaign "
        "--configuration {input.configuration:q} --profile {params.index} --split {wildcards.split} "
        "--shard {params.shard} --device cpu --output {output.report:q}"


rule inference_mean_shard:
    input:
        parent="data/generated/inference/confirmatory-v1/primary-{split}-s{shard}.json",
        arrays="data/generated/inference/confirmatory-v1/primary-{split}-s{shard}.pt",
        configuration="configs/inference/mean_v1.json",
        source="analysis/learning_rule_inference/mean_response.py",
        scientific_sources=INFERENCE_SOURCES,
    output:
        report="data/generated/inference/mean-v1/primary-{split}-s{shard}.json",
        arrays="data/generated/inference/mean-v1/primary-{split}-s{shard}.pt",
    wildcard_constraints:
        split="calibration|evaluation",
        shard="[0-9]{2}",
    params:
        task=lambda w:int(w.shard)+(0 if w.split=="calibration" else 10),
    threads: 1
    resources:
        mem_mb=5632,
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.mean_response "
        "--configuration {input.configuration:q} --task-index {params.task} "
        "--input-dir data/generated/inference/confirmatory-v1 --output-dir data/generated/inference/mean-v1"


rule inference_evaluation:
    input:
        reports=INFERENCE_REPORTS,
        arrays=[p.replace('.json','.pt') for p in INFERENCE_REPORTS],
        configuration="configs/inference/confirmatory_v1.json",
        source="analysis/learning_rule_inference/evaluate.py",
    output:
        "results/data/inference_evaluation.csv",
        "results/data/inference_paired_comparison.csv",
        "results/data/inference_loss_matched_stages.csv",
        "results/data/inference_evaluation.provenance.json",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.evaluate "
        "--configuration {input.configuration:q} --input-dir data/generated/inference/confirmatory-v1"


rule inference_mean_evaluation:
    input:
        reports=MEAN_REPORTS,
        arrays=[p.replace('.json','.pt') for p in MEAN_REPORTS],
        source="analysis/learning_rule_inference/evaluate_mean.py",
        configuration="configs/inference/mean_v1.json",
        parent_configuration="configs/inference/confirmatory_v1.json",
    output:
        "results/data/inference_mean_evaluation.csv",
        "results/data/inference_mean_accuracy.csv",
        "results/data/inference_mean_evaluation.provenance.json",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.evaluate_mean "
        "--input-dir data/generated/inference/mean-v1"


rule inference_original_baseline:
    input:
        source="analysis/learning_rule_inference/baseline.py",
        original="analysis/generate_main_figures/run.py",
        diagnostics="analysis/learning_rule_inference/diagnostics.py",
    output:
        "results/data/inference_original_baseline.csv",
        "results/data/inference_original_baseline.provenance.json",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.baseline "
        "--output results/data/inference_original_baseline.csv"


rule inference_development_checkpoint:
    input:
        configuration="configs/inference/development.json",
        sources=INFERENCE_SOURCES,
    output:
        report="data/generated/inference/matched-development-p0.json",
        arrays="data/generated/inference/matched-development-p0.pt",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.campaign "
        "--configuration {input.configuration:q} --profile 0 --split development --shard 0 "
        "--device cpu --output {output.report:q}"


rule inference_step_mechanism:
    input:
        source="analysis/learning_rule_inference/run.py",
        sources=INFERENCE_SOURCES,
    output:
        "data/generated/inference/pilot-q8-trained.json",
        "data/generated/inference/pilot-q8-trained.pt",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.run pilot "
        "--device cpu --outputs 8 --width 32 --depth 2 --networks 20 --seed-start 100000 "
        "--train-steps 1000 --sigma .001 --output data/generated/inference/pilot-q8-trained.json"


rule inference_shared_sensitivity:
    input:
        parent="data/generated/inference/matched-development-p0.json",
        arrays="data/generated/inference/matched-development-p0.pt",
        source="analysis/learning_rule_inference/sensitivity.py",
    output:
        "data/generated/inference/sensitivity-development.json",
        "data/generated/inference/sensitivity-development.pt",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.sensitivity"


rule inference_publication:
    input:
        "results/data/inference_evaluation.csv",
        "results/data/inference_evaluation.provenance.json",
        "results/data/inference_paired_comparison.csv",
        "results/data/inference_original_baseline.csv",
        "results/data/inference_original_baseline.provenance.json",
        "results/data/inference_mean_evaluation.csv",
        "results/data/inference_mean_accuracy.csv",
        "results/data/inference_mean_evaluation.provenance.json",
        "results/data/inference_gpu_evaluation.csv",
        "results/data/inference_gpu_evaluation.provenance.json",
        "results/data/inference_backend_validation.csv",
        "results/data/inference_backend_validation.provenance.json",
        "results/data/inference_cuda_replication_validation.json",
        "data/generated/inference/pilot-q8-trained.json",
        "data/generated/inference/sensitivity-development.json",
        "data/generated/inference/mechanism-controls-development.json",
        "data/generated/inference/fit-sensitivity-development.json",
        "analysis/learning_rule_inference/publication.py",
    output:
        INFERENCE_FIGURES,
        INFERENCE_TABLES,
        "results/data/inference_prediction_summary.csv",
        "results/data/inference_layerwise_summary.csv",
        "results/data/inference_mechanism_controls.csv",
        "results/data/inference_scale_sensitivity.csv",
        "results/data/inference_fit_sensitivity.csv",
        "results/data/inference_publication.provenance.json",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.publication"


rule collect_inference_assets:
    input:
        figures=[f"results/figures/{n}.pdf" for n in INFERENCE_FIGURE_NAMES],
        tables=INFERENCE_TABLES + ["results/tables/inference_hardware.tex", "results/tables/inference_hardware_b200.tex"],
        hardware="results/data/inference_hardware_benchmark.provenance.json",
        hardware_data="results/data/inference_hardware_benchmark.csv",
        full_hardware="results/data/inference_hardware_b200.provenance.json",
        full_hardware_data="results/data/inference_hardware_b200.csv",
        collector="tools/collect_inference_assets.py",
    output:
        INFERENCE_PAPER_FIGURES,
        INFERENCE_PAPER_TABLES,
        f"{INFERENCE_STAGING}/analysis_provenance.json",
    shell:
        "python tools/collect_inference_assets.py"


rule inference_hardware:
    input:
        configuration="configs/inference/hardware_mig45_v1.json",
        full_configuration="configs/inference/hardware_b200_v1.json",
        reports=HARDWARE_INPUTS,
        source="tools/summarize_inference_hardware.py",
    output:
        "results/data/inference_hardware_benchmark.csv",
        "results/data/inference_hardware_benchmark.provenance.json",
        "results/tables/inference_hardware.tex",
        "results/data/inference_hardware_b200.csv",
        "results/data/inference_hardware_b200.provenance.json",
        "results/tables/inference_hardware_b200.tex",
    shell:
        """
        python tools/summarize_inference_hardware.py
        python tools/summarize_inference_hardware.py --configuration configs/inference/hardware_b200_v1.json --output results/data/inference_hardware_b200.csv --table results/tables/inference_hardware_b200.tex
        """


rule inference_mechanism_controls:
    input:
        parent="data/generated/inference/matched-development-p0.json",
        arrays="data/generated/inference/matched-development-p0.pt",
        configuration="configs/inference/mechanism_controls_v1.json",
        source="analysis/learning_rule_inference/mechanism_controls.py",
    output:
        "data/generated/inference/mechanism-controls-development.json",
        "data/generated/inference/mechanism-controls-development.pt",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.mechanism_controls"


rule inference_fit_sensitivity:
    input:
        parent="data/generated/inference/matched-development-p0.json",
        arrays="data/generated/inference/matched-development-p0.pt",
        configuration="configs/inference/fit_sensitivity_v1.json",
        source="analysis/learning_rule_inference/fit_sensitivity.py",
    output:
        "data/generated/inference/fit-sensitivity-development.json",
        "data/generated/inference/fit-sensitivity-development.pt",
    shell:
        "PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.fit_sensitivity"
