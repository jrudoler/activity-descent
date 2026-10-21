"""Compile the independent manuscript and record source/asset/PDF checksums."""

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
    parser.add_argument("--compiler", choices=("auto", "tectonic", "latexmk"), default="auto")
    parser.add_argument("--cache-directory", type=Path, default=Path("/tmp/activity-descent-tex-cache"))
    args = parser.parse_args()
    paper = args.paper.resolve()
    baseline_path = paper / "manuscript_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        for name, wanted in baseline["protected_files"].items():
            if sha(paper / name) != wanted:
                raise RuntimeError(f"Preserved manuscript changed: {name}. Research belongs in notes or separate appendices until the user approves promotion.")
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
    compatibility = []
    original_source = (paper / "activity_dynamics.tex").read_text()
    watermark = "hpos=300px,vpos=70px"
    if choice == "tectonic" and watermark in original_source:
        # XeTeX does not accept pdfTeX's px unit. Preserve the canonical source;
        # apply this formatting-only compatibility change in a temporary tree.
        with tempfile.TemporaryDirectory(prefix="activity-descent-original-") as temporary:
            staged = Path(temporary)
            compiled_source = original_source.replace(watermark, "hpos=300bp,vpos=70bp")
            (staged / "activity_dynamics.tex").write_text(compiled_source)
            for source in [paper / "activity_dynamics.bib", paper / "generated",
                           *paper.glob("*.cls"), *paper.glob("*.sty")]:
                (staged / source.name).symlink_to(source, target_is_directory=source.is_dir())
            subprocess.run(command, cwd=staged, env=env, check=True)
            for result in staged.glob("activity_dynamics.*"):
                if result.suffix not in (".tex", ".bib") and not result.is_symlink():
                    shutil.copy2(result, paper / result.name)
            compatibility.append({"operation": "legacy watermark px units replaced by bp in a temporary XeTeX input only",
                                  "canonical_source_sha256": sha(paper / "activity_dynamics.tex"),
                                  "compiled_source_sha256": hashlib.sha256(compiled_source.encode()).hexdigest()})
    else:
        subprocess.run(command, cwd=paper, env=env, check=True)
    log = (paper / "activity_dynamics.log").read_text(errors="replace")
    unresolved = [line for line in log.splitlines() if "undefined" in line.lower() and
                  ("citation" in line.lower() or "reference" in line.lower())]
    if unresolved:
        raise RuntimeError("Unresolved references after compilation: " + "; ".join(unresolved))
    sources = [paper / "activity_dynamics.tex", paper / "activity_dynamics.bib", paper / "jmlr.cls", paper / "jmlrutils.sty"]
    sources += list((paper / "generated").glob("*.tex"))
    sources += list((paper / "generated/figures").glob("*.pdf"))
    if baseline_path.exists():
        sources.append(baseline_path)
    payload = {"built_utc": datetime.now(timezone.utc).isoformat(), "compiler": version,
               "build_script_sha256": sha(Path(__file__)),
               "source_date_epoch": env["SOURCE_DATE_EPOCH"],
               "build_compatibility": compatibility,
               "sources": {str(p.relative_to(paper)):sha(p) for p in sorted(sources)},
               "pdf_sha256": sha(paper / "activity_dynamics.pdf"),
               "unresolved_references": unresolved}
    (paper / "generated/build_provenance.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(f"Compiled manuscript: {paper / 'activity_dynamics.pdf'}")


if __name__ == "__main__":
    main()
