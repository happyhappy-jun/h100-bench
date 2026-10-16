"""Retain source, package versions, host samples and a checksum inventory per run."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def preserve_provenance(root: Path, output: Path):
    destination = output / "source"
    destination.mkdir()
    for name in ("benchkit", "configs", "scripts", "templates", "docs"):
        shutil.copytree(root / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "AGENTS.md", "requirements-report.txt"):
        shutil.copy2(root / name, destination / name)
    packages = sorted(({"name": d.metadata["Name"], "version": d.version}
                       for d in importlib.metadata.distributions()), key=lambda x: x["name"].lower())
    dump(output / "packages.json", packages)
    # Deliberately record names/versions, not direct URLs or the complete process
    # environment: those can contain model, cloud and package-index credentials.
    names = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
             "HIP_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NCCL_DEBUG",
             "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "CUDA_MODULE_LOADING",
             "SLURM_JOB_ID", "SLURM_JOB_QOS", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS")
    dump(output / "runtime_environment.json", {k: os.environ[k] for k in names if k in os.environ})


def inventory(output: Path, max_hash_bytes=None):
    """Hash retained files, including checkpoints, without following symlinks."""
    started = time.monotonic()
    entries, errors = [], []
    for path in sorted(output.rglob("*")):
        if path.name in {"artifact_inventory.json", "artifact_inventory.json.tmp"}:
            continue
        try:
            relative = str(path.relative_to(output))
            if path.is_symlink():
                entries.append({"path": relative, "kind": "symlink", "target": os.readlink(path)})
            elif path.is_file():
                size = path.stat().st_size
                if max_hash_bytes is not None and size > max_hash_bytes:
                    entries.append({"path": relative, "kind": "file", "bytes": size,
                                    "sha256": None, "hash_status": "deferred_size_limit"})
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                entries.append({"path": relative, "kind": "file", "bytes": path.stat().st_size,
                                "sha256": digest.hexdigest()})
        except OSError as exc:
            errors.append({"path": str(path.relative_to(output)), "error": str(exc)})
    result = {"schema_version": 1, "files": entries,
              "file_count": sum(x["kind"] == "file" for x in entries),
              "symlink_count": sum(x["kind"] == "symlink" for x in entries),
              "logical_bytes": sum(x.get("bytes", 0) for x in entries),
              "hash_elapsed_s": time.monotonic() - started, "errors": errors,
              "max_hash_bytes": max_hash_bytes,
              "deferred_hash_count": sum(x.get("hash_status") == "deferred_size_limit" for x in entries),
              "scope": "post-run retained files; external symlink targets excluded; not physical disk allocation"}
    dump(output / "artifact_inventory.json", result)
    return result


class HostMonitor:
    """One-second CPU/RAM/I/O snapshots of this process tree; no command lines."""
    def __init__(self, output: Path):
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _sample(self):
        with (self.output / "host_process.jsonl").open("w") as stream:
            try:
                import psutil
                root = psutil.Process()
                while not self.stop_event.is_set():
                    processes = []
                    for process in [root] + root.children(recursive=True):
                        try:
                            with process.oneshot():
                                record = {"pid": process.pid, "name": process.name(),
                                          "memory": process.memory_info()._asdict(),
                                          "cpu_times": process.cpu_times()._asdict(),
                                          "threads": process.num_threads()}
                                if hasattr(process, "io_counters"):
                                    record["io"] = process.io_counters()._asdict()
                                processes.append(record)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    stream.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "host_memory": psutil.virtual_memory()._asdict(),
                        "load_average": os.getloadavg(), "processes": processes}) + "\n")
                    stream.flush()
                    self.stop_event.wait(1)
            except Exception as exc:
                stream.write(json.dumps({"telemetry_error": f"{type(exc).__name__}: {exc}"}) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild a retained run's complete file hash inventory after the timed campaign")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not (args.run_dir / "manifest.json").is_file():
        parser.error("run_dir must contain manifest.json")
    result = inventory(args.run_dir.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False))
