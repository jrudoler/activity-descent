"""Compile the independent manuscript and record source/asset/PDF checksums."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    parser.add_argument("--compiler", choices=("auto", "tectonic", "latexmk"), default="auto")
    parser.add_argument("--cache-directory", type=Path, default=Path("/tmp/activity-descent-tex-cache"))
    args = parser.parse_args()
    paper = args.paper.resolve()
    tectonic = str(ROOT / ".tools/tectonic") if (ROOT / ".tools/tectonic").is_file() else shutil.which("tectonic")
    choice = args.compiler
    if choice == "auto":
        choice = "latexmk" if shutil.which("latexmk") else "tectonic"
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(args.cache_directory.resolve())
    env["SOURCE_DATE_EPOCH"] = "1792368000"
    if choice == "tectonic":
        if not tectonic:
            raise RuntimeError("Install the pinned compiler first: python tools/install_tectonic.py")
        version = subprocess.check_output([tectonic, "--version"], text=True).strip()
        command = [tectonic, "--keep-logs", "--keep-intermediates", "activity_dynamics.tex"]
    else:
        version = subprocess.check_output(["latexmk", "--version"], text=True).strip()
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "activity_dynamics.tex"]
    subprocess.run(command, cwd=paper, env=env, check=True)
    log = (paper / "activity_dynamics.log").read_text(errors="replace")
    unresolved = [line for line in log.splitlines() if "undefined" in line.lower() and
                  ("citation" in line.lower() or "reference" in line.lower())]
    if unresolved:
        raise RuntimeError("Unresolved references after compilation: " + "; ".join(unresolved))
    sources = [paper / "activity_dynamics.tex", paper / "activity_dynamics.bib", paper / "jmlr.cls", paper / "jmlrutils.sty"]
    sources += list((paper / "generated").glob("inference_*.tex"))
    sources += list((paper / "generated/figures").glob("fig_inference_*.pdf"))
    if (paper / "generated/analysis_provenance.json").exists():
        sources.append(paper / "generated/analysis_provenance.json")
    payload = {"built_utc": datetime.now(timezone.utc).isoformat(), "compiler": version,
               "build_script_sha256": sha(Path(__file__)),
               "source_date_epoch": env["SOURCE_DATE_EPOCH"],
               "sources": {str(p.relative_to(paper)):sha(p) for p in sorted(sources)},
               "pdf_sha256": sha(paper / "activity_dynamics.pdf"),
               "unresolved_references": unresolved}
    (paper / "generated/build_provenance.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(f"Compiled manuscript: {paper / 'activity_dynamics.pdf'}")


if __name__ == "__main__":
    main()
