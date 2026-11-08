"""Only the run artifacts needed to reproduce a minimal experiment."""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .util import write_json, write_jsonl


def write_run(run_root: Path, run_id: str, *, manifest: dict[str, Any], rankings: list[dict[str, Any]], metrics: dict[str, Any], latency_cost: list[dict[str, Any]], provider_ledger: list[dict[str, Any]]) -> Path:
    destination = run_root / run_id
    if destination.exists():
        raise ValueError(f"run directory already exists: {destination}")
    run_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=run_root))
    try:
        manifest = {**manifest, "completed_at": datetime.now(UTC).isoformat()}
        write_json(staging / "manifest.json", manifest)
        write_jsonl(staging / "rankings.jsonl", rankings)
        write_json(staging / "metrics.json", metrics)
        write_jsonl(staging / "latency_cost.jsonl", latency_cost)
        write_jsonl(staging / "provider_ledger.jsonl", provider_ledger)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
