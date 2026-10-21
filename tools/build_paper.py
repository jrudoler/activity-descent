"""Build the unchanged manuscript with its specified pdfLaTeX engine."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    parser.add_argument("--compiler", choices=("auto", "latexmk"), default="auto")
    parser.add_argument("--cache-directory", type=Path, default=Path("/tmp/activity-descent-tex-cache"))
    args = parser.parse_args()
    paper = args.paper.resolve()
    baseline_path = paper / "manuscript_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        for name, wanted in baseline["protected_files"].items():
            if sha(paper / name) != wanted:
                raise RuntimeError(f"Preserved manuscript changed: {name}. Research belongs in notes or separate appendices until the user approves promotion.")
    env = os.environ.copy()
    local_texbin = ROOT / ".tools/texlive/bin/x86_64-linux"
    if local_texbin.is_dir():
        env["PATH"] = str(local_texbin) + os.pathsep + env.get("PATH", "")
    env["XDG_CACHE_HOME"] = str(args.cache_directory.resolve())
    env["SOURCE_DATE_EPOCH"] = "1792368000"
    latexmk = shutil.which("latexmk", path=env["PATH"])
    pdflatex = shutil.which("pdflatex", path=env["PATH"])
    if not latexmk or not pdflatex:
        raise RuntimeError("Install TeX Live with pdfLaTeX and latexmk (on PATH or under .tools/texlive). "
                           "The original manuscript requires pdfLaTeX; no substitute engine or source rewrite is used.")
    version = subprocess.check_output([pdflatex, "--version"], env=env, text=True).splitlines()[0]
    driver_version = subprocess.check_output([latexmk, "--version"], env=env, text=True).strip()
    command = [latexmk, "-norc", "-r", "latexmkrc", "-pdf", "-interaction=nonstopmode",
               "-halt-on-error", "-no-shell-escape", "activity_dynamics.tex"]
    # Start without old auxiliary files. Copy exact source bytes; in particular,
    # never change watermark units, font settings, or document class options.
    with tempfile.TemporaryDirectory(prefix="activity-descent-original-") as temporary:
        staged = Path(temporary)
        for source in [paper / "activity_dynamics.tex", paper / "activity_dynamics.bib", paper / "latexmkrc",
                       *paper.glob("*.cls"), *paper.glob("*.sty"), *paper.glob("*.bst")]:
            shutil.copy2(source, staged / source.name)
        shutil.copytree(paper / "generated", staged / "generated")
        subprocess.run(command, cwd=staged, env=env, check=True)
        log = (staged / "activity_dynamics.log").read_text(errors="replace")
        if not log.startswith("This is pdfTeX"):
            raise RuntimeError("The preserved manuscript must be compiled with pdfLaTeX.")
        unresolved = [line for line in log.splitlines() if "undefined" in line.lower() and
                      ("citation" in line.lower() or "reference" in line.lower())]
        if unresolved:
            raise RuntimeError("Unresolved references after compilation: " + "; ".join(unresolved))
        if (staged / "activity_dynamics.tex").read_bytes() != (paper / "activity_dynamics.tex").read_bytes():
            raise RuntimeError("Compiled source differs from the preserved manuscript.")
        for suffix in (".pdf", ".log", ".bbl", ".blg"):
            result = staged / ("activity_dynamics" + suffix)
            if result.is_file():
                shutil.copy2(result, paper / result.name)
    sources = [paper / "activity_dynamics.tex", paper / "activity_dynamics.bib", paper / "jmlr.cls",
               paper / "jmlrutils.sty", paper / "latexmkrc"]
    sources += list((paper / "generated").glob("*.tex"))
    sources += list((paper / "generated/figures").glob("*.pdf"))
    if baseline_path.exists():
        sources.append(baseline_path)
    payload = {"built_utc": datetime.now(timezone.utc).isoformat(), "compiler": version,
               "build_driver": driver_version, "engine": "pdflatex",
               "build_script_sha256": sha(Path(__file__)),
               "source_date_epoch": env["SOURCE_DATE_EPOCH"],
               "build_compatibility": [],
               "compiled_source_sha256": sha(paper / "activity_dynamics.tex"),
               "sources": {str(p.relative_to(paper)):sha(p) for p in sorted(sources)},
               "pdf_sha256": sha(paper / "activity_dynamics.pdf"),
               "unresolved_references": unresolved}
    (paper / "generated/build_provenance.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(f"Compiled manuscript: {paper / 'activity_dynamics.pdf'}")


if __name__ == "__main__":
    main()
