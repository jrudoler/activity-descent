"""Verify archived CPU/CUDA preflights and summarize matched hardware timings."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path,
                        default=Path("configs/inference/hardware_mig45_v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/data/inference_hardware_benchmark.csv"))
    parser.add_argument("--table", type=Path,
                        default=Path("results/tables/inference_hardware.tex"))
    args = parser.parse_args()
    configuration = ROOT / args.configuration
    config = json.loads(configuration.read_text())
    directory = ROOT / config["input_directory"]
    inputs = {}

    def read(name):
        path = directory / name
        inputs[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def source_signature(report):
        sources = report["provenance"]["sources"]
        return {name: sources[f"analysis/learning_rule_inference/{name}.py"]
                for name in config["required_sources"]}

    validations = {}
    for device, job in (("cpu", config["cpu_job_id"]), ("cuda", config["gpu_job_id"])):
        report = read(f"validation-{device}-{job}.json")
        require(report["passed"] and report["tests"] == 18 and not report["failures"],
                f"The {device} scientific preflight did not pass all 18 checks")
        require(report["provenance"]["slurm_job_id"] == job
                and report["provenance"]["device"] == device,
                f"Unexpected {device} validation provenance")
        validations[device] = report
    signature = source_signature(validations["cuda"])
    require(source_signature(validations["cpu"]) == signature,
            "CPU and CUDA validation used different implementations")
    reference_runtime = validations["cuda"]["provenance"]
    runtime_keys = ("python", "torch", "cuda_runtime", "matmul_precision", "lockfile_sha256")
    require(all(validations["cpu"]["provenance"][k] == reference_runtime[k] for k in runtime_keys),
            "CPU and CUDA validation used different runtime environments")
    require(reference_runtime.get("gpu") == config["gpu_name"],
            "CUDA validation ran on an unexpected GPU")

    rows, devices = [], {}
    for networks in config["batch_sizes"]:
        reports = {}
        for device, job in (("cpu", config["cpu_job_id"]), ("cuda", config["gpu_job_id"])):
            report = read(f"benchmark-{device}-n{networks}-{job}.json")
            provenance = report["provenance"]
            require(provenance["device"] == device and provenance["slurm_job_id"] == job,
                    "Unexpected benchmark device or job identity")
            require(source_signature(report) == signature,
                    "Benchmark and scientific validation implementations differ")
            require(all(provenance[k] == reference_runtime[k] for k in runtime_keys),
                    "Benchmarks and validation used different runtime environments")
            require(provenance["threads"] == 1 and report["configuration"]["threads"] == 1,
                    "This comparison requires single-thread CPU operation")
            require(report["configuration"]["mode"] == "benchmark"
                    and report["configuration"]["networks"] == networks,
                    "Unexpected benchmark workload")
            require(all(report["configuration"][k] == v for k, v in config["workload"].items()),
                    "Benchmark settings differ from the archived hardware protocol")
            if device == "cuda":
                require(provenance.get("gpu") == config["gpu_name"],
                        "The report does not describe the configured B200 MIG slice")
            devices[device] = {k: provenance.get(k) for k in
                ("host", "python", "torch", "cuda_runtime", "threads", "gpu",
                 "gpu_capability", "gpu_total_memory", "driver", "slurm_job_id")}
            reports[device] = report
        workload = lambda d: {k: v for k, v in d["configuration"].items()
                              if k not in ("device", "output")}
        require(workload(reports["cpu"]) == workload(reports["cuda"])
                and reports["cpu"]["seeds"] == reports["cuda"]["seeds"],
                "CPU and CUDA workloads or initial seed identities differ")
        indexed = {}
        for device, report in reports.items():
            entries = {(r["dtype"], r["rule"]): r for r in report["timings"]}
            require(len(entries) == len(report["timings"]) == 6,
                    "Incomplete or duplicated dtype/rule timings")
            indexed[device] = entries
        require(indexed["cpu"].keys() == indexed["cuda"].keys(), "Timing conditions differ")
        for key, cpu in indexed["cpu"].items():
            gpu = indexed["cuda"][key]
            require(all(cpu[k] == gpu[k] for k in
                        ("networks", "probes_per_network", "perturbation_repeats")),
                    "Probe bank dimensions differ")
            rows.append({"networks": networks, "dtype": key[0], "operation": key[1],
                         "timing_scope": "mean_warmed_probe_bank",
                         "repeats": reports["cpu"]["configuration"]["benchmark_repeats"],
                         "perturbation_repeats": cpu["perturbation_repeats"],
                         "probes_per_network": cpu["probes_per_network"],
                         "cpu_seconds": cpu["seconds_per_bank"],
                         "gpu_seconds": gpu["seconds_per_bank"],
                         "cpu_over_gpu": cpu["seconds_per_bank"] / gpu["seconds_per_bank"],
                         "cpu_float32_response_error_max": cpu.get("response_error_vs_float64_max", ""),
                         "gpu_float32_response_error_max": gpu.get("response_error_vs_float64_max", "")})
        cpu_ops = {r["operation"]: r for r in reports["cpu"]["inference_timings"]}
        gpu_ops = {r["operation"]: r for r in reports["cuda"]["inference_timings"]}
        require(cpu_ops.keys() == gpu_ops.keys() == {"compatibility", "psd_fit"},
                "Incomplete auxiliary operator timings")
        for operation, cpu in cpu_ops.items():
            gpu = gpu_ops[operation]
            require(cpu["iterations"] == gpu["iterations"] == 500,
                    "Generic PSD solver iteration budgets differ")
            rows.append({"networks": networks, "dtype": "torch.float64", "operation": operation,
                         "timing_scope": "single_auxiliary_call", "repeats": 1,
                         "perturbation_repeats": "", "probes_per_network": "",
                         "cpu_seconds": cpu["seconds"], "gpu_seconds": gpu["seconds"],
                         "cpu_over_gpu": cpu["seconds"] / gpu["seconds"],
                         "cpu_float32_response_error_max": "", "gpu_float32_response_error_max": ""})
    require(all(math.isfinite(r[k]) and r[k] > 0 for r in rows
                for k in ("cpu_seconds", "gpu_seconds")), "Invalid elapsed time")

    accounting = read(config["accounting"])
    allocations = {}
    for job in (config["cpu_job_id"], config["gpu_job_id"]):
        record = accounting["jobs"][job]
        require(record["State"] == "COMPLETED" and record["ExitCode"] == "0:0",
                "The timing allocation did not complete successfully")
        start = datetime.fromisoformat(record["Start"])
        submit = datetime.fromisoformat(record["Submit"])
        eligible = datetime.fromisoformat(record["Eligible"])
        allocations[job] = {**record, "submit_to_start_seconds": (start - submit).total_seconds(),
                            "eligible_to_start_seconds": (start - eligible).total_seconds()}

    output, table = ROOT / args.output, ROOT / args.table
    output.parent.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{lr rrr}", r"\hline",
             r"Operation & Networks & CPU (ms) & MIG (ms) & CPU/MIG \\", r"\hline"]
    names = {"gd": "GD bank", "feedback": "Feedback bank", "node": "Node bank ($R=32$)",
             "compatibility": "Compatibility call", "psd_fit": "Generic PSD fit (500 iter.)"}
    for row in rows:
        if row["dtype"] != "torch.float64":
            continue
        if row["timing_scope"] == "single_auxiliary_call" and row["networks"] != 20:
            continue
        lines.append(f"{names[row['operation']]} & {row['networks']} & "
                     f"{1000*row['cpu_seconds']:.2f} & {1000*row['gpu_seconds']:.2f} & "
                     f"{row['cpu_over_gpu']:.3f} " + r"\\")
    lines += [r"\hline", r"\end{tabular}"]
    table.write_text("\n".join(lines) + "\n")
    payload = {"configuration_sha256": digest(configuration), "script_sha256": digest(Path(__file__)),
               "inputs": inputs, "matched_sources": signature, "devices": devices,
               "allocation_timezone": accounting["timezone"], "allocations": allocations,
               "output_sha256": digest(output), "table_sha256": digest(table),
               "interpretation": "One allocation per backend. Probe times average five warmed repetitions; operator times are single calls. The generic 500-iteration fit is not the main study's analytic span fit. Queue waits describe these jobs only. CUDA validation is distinct from the pending saved-checkpoint backend comparison and scientific replication."}
    output.with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Verified matched CPU/MIG preflights; wrote {len(rows)} timing conditions and {table}")


if __name__ == "__main__":
    main()
