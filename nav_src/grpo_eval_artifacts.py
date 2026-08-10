"""Immutable LoRA snapshots, a resumable evaluation queue, and best selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from action_plan_cache import canonical_json, sha256_file, sha256_text
from lora_policy import ADAPTER_MANIFEST_NAME, validate_local_adapter_directory
from r2r_evaluation import selection_key


SNAPSHOT_MANIFEST_NAME = "navgpt_eval_snapshot.json"
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    ADAPTER_MANIFEST_NAME,
)


class EvaluationArtifactError(RuntimeError):
    """Raised when persisted validation state is incomplete or inconsistent."""


@dataclass(frozen=True)
class EvaluationSnapshot:
    step: int
    path: str
    fingerprint: str
    weights_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "path": self.path,
            "fingerprint": self.fingerprint,
            "weights_sha256": self.weights_sha256,
        }


class EvaluationSnapshotStore:
    """Atomically copy the three LoRA inference files before checkpoint rotation."""

    def __init__(
        self,
        root: str,
        *,
        policy_config: Any,
        run_fingerprint: str,
        validation_fingerprint: str,
        adapter_validator: Optional[Callable[[str], Path]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.policy_config = policy_config
        self.run_fingerprint = str(run_fingerprint)
        self.validation_fingerprint = str(validation_fingerprint)
        self.adapter_validator = adapter_validator or (
            lambda value: validate_local_adapter_directory(value, policy_config)
        )

    def create(self, source_path: str, *, step: int) -> EvaluationSnapshot:
        if step < 0:
            raise ValueError("Evaluation snapshot step must be non-negative")
        source = self.adapter_validator(
            str(Path(source_path).expanduser().resolve())
        )
        source_files = _file_inventory(source, ADAPTER_FILES)
        destination = self.root / f"step-{step}"
        if destination.exists():
            snapshot = self.validate(str(destination), expected_step=step)
            actual = json.loads(
                (destination / SNAPSHOT_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            if actual["files"] != source_files:
                raise EvaluationArtifactError(
                    f"Step {step} snapshot differs from its checkpoint"
                )
            return snapshot

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".step-{step}.", dir=str(self.root))
        )
        try:
            for name in ADAPTER_FILES:
                shutil.copy2(source / name, temporary / name)
            copied_files = _file_inventory(temporary, ADAPTER_FILES)
            if copied_files != source_files:
                raise EvaluationArtifactError(
                    "Eval snapshot copy changed adapter bytes"
                )
            body = {
                "schema_version": 1,
                "snapshot_type": "navgpt_lora_eval_snapshot",
                "run_fingerprint": self.run_fingerprint,
                "validation_fingerprint": self.validation_fingerprint,
                "step": int(step),
                "source_path": str(source),
                "files": copied_files,
            }
            body["snapshot_fingerprint"] = sha256_text(canonical_json(body))
            _write_json(temporary / SNAPSHOT_MANIFEST_NAME, body, exclusive=True)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.validate(str(destination), expected_step=step)

    def validate(
        self,
        snapshot_path: str,
        *,
        expected_step: Optional[int] = None,
    ) -> EvaluationSnapshot:
        path = self.adapter_validator(str(Path(snapshot_path).expanduser().resolve()))
        manifest_path = path / SNAPSHOT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise EvaluationArtifactError(f"Missing eval snapshot manifest: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest.pop("snapshot_fingerprint", None)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("snapshot_type") != "navgpt_lora_eval_snapshot"
            or manifest.get("run_fingerprint") != self.run_fingerprint
            or manifest.get("validation_fingerprint")
            != self.validation_fingerprint
            or fingerprint != sha256_text(canonical_json(manifest))
        ):
            raise EvaluationArtifactError(f"Invalid eval snapshot identity: {path}")
        step = int(manifest.get("step", -1))
        if expected_step is not None and step != expected_step:
            raise EvaluationArtifactError(f"Eval snapshot step changed: {path}")
        if manifest.get("files") != _file_inventory(path, ADAPTER_FILES):
            raise EvaluationArtifactError(f"Eval snapshot files changed: {path}")
        return EvaluationSnapshot(
            step=step,
            path=str(path),
            fingerprint=str(fingerprint),
            weights_sha256=str(
                manifest["files"]["adapter_model.safetensors"]["sha256"]
            ),
        )


class EvaluationQueue:
    """Atomic event/job ledger; a running job is resumed rather than replaced."""

    def __init__(
        self,
        path: str,
        *,
        run_fingerprint: str,
        validation_fingerprint: str,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.run_fingerprint = str(run_fingerprint)
        self.validation_fingerprint = str(validation_fingerprint)

    def initialize(self) -> Dict[str, Any]:
        if self.path.exists():
            return self.read()
        state = {
            "schema_version": 1,
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
            "events": [],
            "jobs": [],
        }
        self._write(state)
        return state

    def read(self) -> Dict[str, Any]:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            state.get("schema_version") != 1
            or state.get("run_fingerprint") != self.run_fingerprint
            or state.get("validation_fingerprint") != self.validation_fingerprint
            or not isinstance(state.get("events"), list)
            or not isinstance(state.get("jobs"), list)
        ):
            raise EvaluationArtifactError("Evaluation queue belongs to another run")
        _unique_rows(state["events"], "event_id")
        _unique_rows(state["jobs"], "job_id")
        return state

    def enqueue_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        required = {
            "event_id",
            "step",
            "source_path",
            "fast_due",
            "epoch_due",
            "epoch",
        }
        if set(event) != required:
            raise EvaluationArtifactError("Evaluation event has unexpected fields")
        state = self.read()
        existing = _find(state["events"], "event_id", str(event["event_id"]))
        if existing is not None:
            immutable = {name: existing[name] for name in required}
            if canonical_json(immutable) != canonical_json(dict(event)):
                raise EvaluationArtifactError("Evaluation event identity changed")
            return existing
        row = {
            **dict(event),
            "status": "queued",
            "snapshot": None,
            "fast_job_id": None,
            "full_candidates": [],
        }
        state["events"].append(row)
        self._write(state)
        return row

    def update_event(self, event_id: str, **updates: Any) -> Dict[str, Any]:
        allowed = {"status", "snapshot", "fast_job_id", "full_candidates"}
        if set(updates).difference(allowed):
            raise EvaluationArtifactError("Unsafe evaluation-event update")
        if "status" in updates and updates["status"] not in {
            "queued",
            "running",
            "completed",
        }:
            raise EvaluationArtifactError("Invalid evaluation-event status")
        state = self.read()
        row = _require(state["events"], "event_id", event_id)
        row.update(updates)
        self._write(state)
        return row

    def pending_events(self) -> Sequence[Dict[str, Any]]:
        return tuple(
            row for row in self.read()["events"] if row["status"] != "completed"
        )

    def enqueue_job(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        required = {
            "job_id",
            "mode",
            "step",
            "snapshot",
            "output_path",
        }
        if set(job) != required or job["mode"] not in {"fast", "full"}:
            raise EvaluationArtifactError("Evaluation job has invalid fields")
        state = self.read()
        existing = _find(state["jobs"], "job_id", str(job["job_id"]))
        if existing is not None:
            immutable = {name: existing[name] for name in required}
            if canonical_json(immutable) != canonical_json(dict(job)):
                raise EvaluationArtifactError("Evaluation job identity changed")
            return existing
        row = {**dict(job), "status": "queued", "result": None}
        state["jobs"].append(row)
        self._write(state)
        return row

    def job(self, job_id: str) -> Dict[str, Any]:
        return _require(self.read()["jobs"], "job_id", job_id)

    def mark_running(self, job_id: str) -> Dict[str, Any]:
        state = self.read()
        row = _require(state["jobs"], "job_id", job_id)
        if row["status"] == "completed":
            return row
        row["status"] = "running"
        self._write(state)
        return row

    def mark_completed(
        self, job_id: str, result: Mapping[str, Any]
    ) -> Dict[str, Any]:
        state = self.read()
        row = _require(state["jobs"], "job_id", job_id)
        if row["status"] == "completed":
            if canonical_json(row["result"]) != canonical_json(dict(result)):
                raise EvaluationArtifactError("Completed evaluation result changed")
            return row
        row["status"] = "completed"
        row["result"] = dict(result)
        self._write(state)
        return row

    def _write(self, state: Mapping[str, Any]) -> None:
        _write_json(self.path, state)


class BestSelector:
    """Select quick/full best exclusively from completed queue results."""

    def __init__(
        self,
        path: str,
        *,
        run_fingerprint: str,
        validation_fingerprint: str,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.run_fingerprint = str(run_fingerprint)
        self.validation_fingerprint = str(validation_fingerprint)

    def initialize(self) -> Dict[str, Any]:
        if self.path.exists():
            return self.read()
        state = {
            "schema_version": 2,
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
            "quick_best": None,
            "full_best": None,
            "fast_history": [],
            "epoch_history": [],
        }
        self._write(state)
        return state

    def read(self) -> Dict[str, Any]:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            state.get("schema_version") != 2
            or state.get("run_fingerprint") != self.run_fingerprint
            or state.get("validation_fingerprint") != self.validation_fingerprint
        ):
            raise EvaluationArtifactError("Best-selector state belongs to another run")
        return state

    def record_fast(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        _require_completed_job(job, "fast")
        state = self.read()
        if any(row["job_id"] == job["job_id"] for row in state["fast_history"]):
            return state
        candidate = _candidate(job, roles=("quick_best",))
        previous = state["quick_best"]
        improved = previous is None or _candidate_key(candidate) > _candidate_key(
            previous
        )
        if improved:
            state["quick_best"] = candidate
        state["fast_history"].append(
            {
                "job_id": job["job_id"],
                "step": int(job["step"]),
                "metrics": candidate["metrics"],
                "improved": improved,
            }
        )
        self._write(state)
        return state

    def record_epoch(
        self,
        *,
        event_id: str,
        step: int,
        epoch: Any,
        candidates: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        state = self.read()
        if any(row["event_id"] == event_id for row in state["epoch_history"]):
            return state
        choices = [dict(value) for value in candidates]
        if not choices:
            raise EvaluationArtifactError("Epoch selector received no candidates")
        previous = state.get("full_best")
        if previous is not None and not any(
            value["snapshot_fingerprint"] == previous["snapshot_fingerprint"]
            for value in choices
        ):
            choices.append({**dict(previous), "roles": ["previous_full_best"]})
        winner = max(choices, key=_candidate_key)
        state["full_best"] = winner
        state["epoch_history"].append(
            {
                "event_id": event_id,
                "step": int(step),
                "epoch": epoch,
                "candidate_job_ids": [
                    value.get("job_id") for value in choices if value.get("job_id")
                ],
                "winner": winner,
            }
        )
        self._write(state)
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        _write_json(self.path, state)


def completed_candidate(
    job: Mapping[str, Any], *, roles: Sequence[str]
) -> Dict[str, Any]:
    _require_completed_job(job, "full")
    return _candidate(job, roles=roles)


def _candidate(job: Mapping[str, Any], *, roles: Sequence[str]) -> Dict[str, Any]:
    snapshot = job["snapshot"]
    result = job["result"]
    return {
        "job_id": str(job["job_id"]),
        "step": int(job["step"]),
        "adapter_path": str(snapshot["path"]),
        "snapshot_fingerprint": str(snapshot["fingerprint"]),
        "evaluation_path": str(job["output_path"]),
        "metrics": dict(result["metrics"]),
        "roles": list(roles),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return selection_key(candidate["metrics"], step=int(candidate["step"]))


def _require_completed_job(job: Mapping[str, Any], mode: str) -> None:
    if job.get("status") != "completed" or job.get("mode") != mode:
        raise EvaluationArtifactError(f"Selector requires a completed {mode} job")
    result = job.get("result")
    if not isinstance(result, Mapping) or not isinstance(
        result.get("metrics"), Mapping
    ):
        raise EvaluationArtifactError("Completed job has no metrics")


def _file_inventory(root: Path, names: Sequence[str]) -> Dict[str, Any]:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise EvaluationArtifactError(f"Adapter files are missing: {missing}")
    return {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in names
    }


def _find(
    rows: Sequence[Dict[str, Any]], key: str, value: str
) -> Optional[Dict[str, Any]]:
    return next((row for row in rows if str(row.get(key)) == value), None)


def _require(rows: Sequence[Dict[str, Any]], key: str, value: str) -> Dict[str, Any]:
    row = _find(rows, key, value)
    if row is None:
        raise EvaluationArtifactError(f"Unknown evaluation record: {value}")
    return row


def _unique_rows(rows: Sequence[Dict[str, Any]], key: str) -> None:
    values = [str(row.get(key, "")) for row in rows]
    if any(not value for value in values) or len(set(values)) != len(values):
        raise EvaluationArtifactError(f"Evaluation ledger has duplicate {key}")


def _write_json(
    path: Path, value: Any, *, exclusive: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as file_obj:
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        file_obj.write(text)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)
