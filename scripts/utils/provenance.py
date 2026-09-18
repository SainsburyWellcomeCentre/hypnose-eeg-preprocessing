"""Record which code and inputs produced a derivative output.

Every stage that writes a derivative (artifact detection, sleep scoring,
quality control) writes a `<output stem>_provenance.json` sidecar beside it
naming the git commit of this checkout, when it ran, and the stage-specific
parameters that determine the result -- for sleep scoring, which model was
used, identified by content hash as well as by path so a replaced model file
is still distinguishable.

The sidecar is written only when the output itself is written: a step skipped
because its output already exists leaves the existing provenance in place, so
the two never disagree about which commit produced which file.

Git state is best-effort. An installed copy of this package, an exported
tarball, or a machine without `git` records `"git": null` rather than failing
the run -- provenance is never a reason for a pipeline to stop.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.io.repository_paths import get_repo_root

PROVENANCE_SCHEMA = "hypnose-eeg-provenance/1"
PROVENANCE_SUFFIX = "_provenance.json"


def _git(*args: str) -> str | None:
    """Run one read-only git command in the repo root, or None if that is not possible."""
    try:
        result = subprocess.run(
            ["git", "-C", str(get_repo_root()), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_revision() -> dict[str, Any] | None:
    """Describe the checkout that is running, or None when there is no git state.

    `dirty` matters more than the commit: a commit hash alone implies the output
    is reproducible from that commit, which is untrue when the working tree had
    uncommitted edits at the time of the run.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git("status", "--porcelain")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "branch": None if branch == "HEAD" else branch,
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status),
    }


def file_fingerprint(path: str | Path) -> dict[str, Any] | None:
    """Identify a file by content, so a replaced file at the same path is visible."""
    path = Path(path)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        stat = path.stat()
    except OSError:
        return None
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
    }


def provenance_record(
    stage: str,
    *,
    outputs: list[str | Path],
    inputs: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance mapping for one stage's outputs."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "python": sys.version.split()[0],
        "command": " ".join(sys.argv),
        "inputs": inputs or {},
        "parameters": parameters or {},
        "outputs": [str(output) for output in outputs],
    }


def provenance_path(output: str | Path) -> Path:
    """The sidecar path for an output, for example `qc_summary_provenance.json`."""
    output = Path(output)
    return output.with_name(f"{output.stem}{PROVENANCE_SUFFIX}")


def write_provenance(
    stage: str,
    *,
    outputs: list[str | Path],
    inputs: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Path:
    """Write the sidecar beside the first output and report it. Never raises."""
    record = provenance_record(
        stage, outputs=outputs, inputs=inputs, parameters=parameters
    )
    path = provenance_path(outputs[0])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    except OSError as exc:
        print(f"Warning: could not write provenance to {path}: {exc}", file=sys.stderr)
        return path
    print(f"Saved: {path}")
    return path
