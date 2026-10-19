"""Runtime and implementation provenance for individual completed jobs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(device):
    device = torch.device(device)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    result = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device), "threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "analysis_commit": command(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(command(["git", "status", "--porcelain"])),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "sources": {str(p.relative_to(ROOT)): digest(p)
                    for p in sorted(Path(__file__).parent.glob("*.py"))},
        "lockfile_sha256": digest(ROOT / "uv.lock"),
    }
    if cuda:
        index = device.index or 0
        result.update({
            "gpu": torch.cuda.get_device_name(index),
            "gpu_capability": list(torch.cuda.get_device_capability(index)),
            "gpu_total_memory": torch.cuda.get_device_properties(index).total_memory,
            "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
        })
    return result


def clean_json(value):
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: clean_json(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
