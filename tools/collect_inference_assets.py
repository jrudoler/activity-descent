"""Copy verified publication assets into the independent manuscript repository."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paper = ROOT / "paper"
    if not (paper / ".git").exists():
        raise RuntimeError("Clone the independent manuscript repository at paper/ before collecting assets")
    data = ROOT / "results/data"
    publication = json.loads((data / "inference_publication.provenance.json").read_text())
    main_report = json.loads((data / "inference_evaluation.provenance.json").read_text())
    mean = json.loads((data / "inference_mean_evaluation.provenance.json").read_text())
    hardware_path = data / "inference_hardware_benchmark.provenance.json"
    hardware = json.loads(hardware_path.read_text())
    if sha(ROOT / "analysis/learning_rule_inference/publication.py") != publication["script_sha256"]:
        raise RuntimeError("Regenerate publication artifacts with the current script first")
    for name, wanted in {**publication["inputs"], **publication["figure_hashes"]}.items():
        if sha(ROOT / name) != wanted:
            raise RuntimeError(f"Publication asset or input changed: {name}")
    hardware_files = {
        **hardware["inputs"],
        "tools/summarize_inference_hardware.py": hardware["script_sha256"],
        "configs/inference/hardware_mig45_v1.json": hardware["configuration_sha256"],
        "results/data/inference_hardware_benchmark.csv": hardware["output_sha256"],
        "results/tables/inference_hardware.tex": hardware["table_sha256"],
    }
    for name, wanted in hardware_files.items():
        if sha(ROOT / name) != wanted:
            raise RuntimeError(f"Hardware evidence or generated asset changed: {name}")
    (paper / "generated/figures").mkdir(parents=True, exist_ok=True)
    for name in publication["figure_hashes"]:
        shutil.copy2(ROOT / name, paper / "generated/figures" / Path(name).name)
    table_names = ("primary", "numbers", "prediction", "regularization", "hardware")
    for name in table_names:
        source = ROOT / "results/tables" / f"inference_{name}.tex"
        shutil.copy2(source, paper / "generated" / source.name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    record = {
        "analysis_commit": commit, "analysis_working_tree_dirty": dirty,
        "main_configuration_sha256": main_report["configuration_sha256"],
        "mean_configuration_sha256": mean["configuration_sha256"],
        "scientific_sources": main_report["science_sources"],
        "publication_script_sha256": publication["script_sha256"],
        "publication_provenance_sha256": sha(data / "inference_publication.provenance.json"),
        "hardware_provenance_sha256": sha(hardware_path),
        "hardware_evidence": hardware,
        "collector_sha256": sha(Path(__file__)),
        "summary_csv_sha256": {p.name:sha(p) for p in sorted(data.glob("inference_*.csv"))},
        "figures": {Path(p).name:h for p,h in publication["figure_hashes"].items()},
        "tables": {p.name:sha(p) for p in sorted((ROOT / "results/tables").glob("inference_*.tex"))},
        "evidence_status": "See the main manuscript and analysis progress record for completed versus queued experiments",
        "preserved_archive_sha256": sha(paper / "archive/neurreps_extended_abstract.tex"),
    }
    (paper / "generated/analysis_provenance.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"Collected {len(publication['figure_hashes'])} figures, {len(table_names)} tables, and analysis provenance at {commit}")


if __name__ == "__main__":
    main()
