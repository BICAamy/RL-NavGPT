"""Minimal in-training Val-Unseen scheduling and best-LoRA selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from action_plan_cache import canonical_json, sha256_file, sha256_text
from grpo_eval_artifacts import (
    BestSelector,
    EvaluationQueue,
    EvaluationSnapshotStore,
    completed_candidate,
)
from r2r_evaluation import (
    NATIVE_EVALUATOR_FAMILY,
    NATIVE_EVALUATOR_SCHEMA_VERSION,
    NativeR2REvaluationService,
    R2REvaluationConfig,
    ResumableEvaluationStore,
    build_resumable_evaluation_manifest,
    build_native_policy_identity,
    load_fast_subset_manifest,
    load_official_native_manifest,
    load_validation_dataset,
    prepare_fast_subset_manifest,
)


class GRPOValidationError(RuntimeError):
    pass


EPOCH_END_FULL_REASON = "epoch_end"
TRAIN_END_FULL_REASON = "train_end"
_FULL_REASONS = {EPOCH_END_FULL_REASON, TRAIN_END_FULL_REASON}


@dataclass(frozen=True)
class GRPOValidationConfig:
    evaluation: R2REvaluationConfig
    fast_subset_manifest: str
    fast_subset_size: int = 128
    fast_subset_seed: int = 0
    fast_interval_steps: int = 1_000
    progress_interval: int = 10

    def __post_init__(self) -> None:
        if min(
            self.fast_subset_size,
            self.fast_interval_steps,
            self.progress_interval,
        ) <= 0:
            raise ValueError("Validation sizes and intervals must be positive")

    @property
    def subset_path(self) -> Path:
        return Path(self.fast_subset_manifest).expanduser().resolve()


def prepare_validation_contract(
    config: Optional[GRPOValidationConfig],
) -> Dict[str, Any]:
    if config is None:
        return {"schema_version": 1, "enabled": False}
    config.evaluation.validate()
    prepare_fast_subset_manifest(
        config.evaluation.annotation,
        str(config.subset_path),
        subset_size=config.fast_subset_size,
        seed=config.fast_subset_seed,
        expected_instruction_count=config.evaluation.expected_instruction_count,
    )
    body = {
        "schema_version": 1,
        "enabled": True,
        "evaluation": config.evaluation.identity(),
        "fast_subset_sha256": sha256_file(config.subset_path),
        "fast_subset_size": config.fast_subset_size,
        "fast_subset_seed": config.fast_subset_seed,
        "fast_interval_steps": config.fast_interval_steps,
        "progress_interval": config.progress_interval,
        "selection": ["spl", "sr", "nDTW", "lower_nav_error", "earlier_step"],
    }
    body["validation_fingerprint"] = sha256_text(canonical_json(body))
    return body


class GRPOValidationManager:
    def __init__(
        self,
        *,
        policy: Any,
        config: GRPOValidationConfig,
        contract: Mapping[str, Any],
        output_dir: str,
        run_fingerprint: str,
        distributed_context: Any,
    ) -> None:
        self.policy = policy
        self.config = config
        self.contract = dict(contract)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.validation_dir = self.output_dir / "validation"
        self.state_path = self.validation_dir / "state.json"
        self.queue_path = self.validation_dir / "queue.json"
        self.run_fingerprint = str(run_fingerprint)
        self.distributed = distributed_context
        dataset = load_validation_dataset(config.evaluation)
        self.dataset = dataset
        self.fast_ids = load_fast_subset_manifest(
            str(config.subset_path), dataset, expected_size=config.fast_subset_size
        )
        self.full_ids = dataset.instr_ids
        self.service = NativeR2REvaluationService(dataset)
        self.protocol = self.distributed.call_on_main_and_broadcast(
            lambda: self.service.protocol(
                self.policy.tokenizer,
                model_path=self.policy.config.model_path,
                dtype=self.policy.config.dtype,
            )
        )
        fingerprint = str(self.contract["validation_fingerprint"])
        self.snapshots = EvaluationSnapshotStore(
            str(self.validation_dir / "snapshots"),
            policy_config=self.policy.config,
            run_fingerprint=self.run_fingerprint,
            validation_fingerprint=fingerprint,
        )
        self.queue = EvaluationQueue(
            str(self.queue_path),
            run_fingerprint=self.run_fingerprint,
            validation_fingerprint=fingerprint,
        )
        self.selector = BestSelector(
            str(self.state_path),
            run_fingerprint=self.run_fingerprint,
            validation_fingerprint=fingerprint,
        )
        self.distributed.call_on_main_and_broadcast(self._initialize_artifacts)
        self.distributed.call_on_main_and_broadcast(self._validate_selector_state)

    def resume_pending(self, *, current_step: int) -> int:
        pending = self.distributed.call_on_main_and_broadcast(
            lambda: list(self.queue.pending_events())
        )
        for event in pending:
            self._full_reason(event)
            if int(event["step"]) != int(current_step):
                raise GRPOValidationError(
                    f"Pending validation is step {event['step']}, not {current_step}"
                )
            self._execute_event(event)
        return len(pending)

    def run_scheduled_checkpoint(
        self,
        *,
        step: int,
        checkpoint_path: str,
        fast_due: bool,
        epoch_due: bool,
        epoch: Optional[float],
        full_reason: str = EPOCH_END_FULL_REASON,
    ) -> None:
        if not fast_due and not epoch_due:
            return
        if epoch_due and full_reason not in _FULL_REASONS:
            raise GRPOValidationError(
                f"Invalid full-validation reason: {full_reason}"
            )
        if not epoch_due and full_reason != EPOCH_END_FULL_REASON:
            raise GRPOValidationError(
                "A full-validation reason requires epoch_due=True"
            )
        if epoch_due and full_reason == TRAIN_END_FULL_REASON:
            event_id = (
                f"step-{int(step)}-fast-{int(bool(fast_due))}-train-end"
            )
        else:
            # Keep the original event identity for epoch scheduling and for
            # fast-only events so existing queues remain resumable.
            event_id = (
                f"step-{int(step)}-fast-{int(bool(fast_due))}"
                f"-epoch-{int(bool(epoch_due))}"
            )
        event = {
            "event_id": event_id,
            "step": int(step),
            "source_path": str(Path(checkpoint_path).resolve()),
            "fast_due": bool(fast_due),
            "epoch_due": bool(epoch_due),
            "epoch": epoch,
        }
        self._full_reason(event)
        queued = self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.enqueue_event(event)
        )
        if queued["status"] != "completed":
            self._execute_event(queued)

    def ensure_train_end_validation(
        self,
        *,
        step: int,
        checkpoint_path: str,
        epoch: Optional[float],
    ) -> None:
        """Run one final full selection, or reuse a same-step full event.

        A max-step boundary can also be an epoch boundary.  In that case the
        epoch callback may already have completed the exact full-validation
        lifecycle required at train end.  Reusing that immutable event avoids
        a second selector decision and prevents evaluation artifacts from
        being assigned two lifecycle identities.
        """

        step = int(step)
        source_path = str(Path(checkpoint_path).resolve())

        def same_step_full_events() -> Sequence[Dict[str, Any]]:
            matches = []
            for event in self.queue.read()["events"]:
                reason = self._full_reason(event)
                if int(event["step"]) == step and reason is not None:
                    matches.append(dict(event))
            return tuple(matches)

        existing = self.distributed.call_on_main_and_broadcast(
            same_step_full_events
        )
        if existing:
            for event in existing:
                if str(Path(str(event["source_path"])).resolve()) != source_path:
                    raise GRPOValidationError(
                        "Same-step full validation references another checkpoint"
                    )
                if event["status"] == "completed":
                    self.distributed.call_on_main_and_broadcast(
                        lambda event=event: self._validate_completed_full_event(
                            event
                        )
                    )
                else:
                    self._execute_event(event)
            return

        self.run_scheduled_checkpoint(
            step=step,
            checkpoint_path=source_path,
            fast_due=False,
            epoch_due=True,
            epoch=epoch,
            full_reason=TRAIN_END_FULL_REASON,
        )

    @staticmethod
    def _full_reason(event: Mapping[str, Any]) -> Optional[str]:
        """Validate a persisted event ID and recover its full lifecycle role."""

        try:
            step = int(event["step"])
            fast_due = bool(event["fast_due"])
            epoch_due = bool(event["epoch_due"])
            event_id = str(event["event_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GRPOValidationError(
                "Validation event identity is incomplete"
            ) from exc
        legacy_id = (
            f"step-{step}-fast-{int(fast_due)}-epoch-{int(epoch_due)}"
        )
        train_end_id = f"step-{step}-fast-{int(fast_due)}-train-end"
        if event_id == legacy_id:
            return EPOCH_END_FULL_REASON if epoch_due else None
        if event_id == train_end_id and epoch_due:
            return TRAIN_END_FULL_REASON
        raise GRPOValidationError(
            f"Validation event ID disagrees with its schedule: {event_id}"
        )

    def _execute_event(self, event: Mapping[str, Any]) -> None:
        self._full_reason(event)
        event_id = str(event["event_id"])
        event = self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.update_event(event_id, status="running")
        )
        snapshot = self.distributed.call_on_main_and_broadcast(
            lambda: self._snapshot_event(event_id)
        )
        if event["fast_due"]:
            fast_job = self.distributed.call_on_main_and_broadcast(
                lambda: self._enqueue_job(snapshot, "fast")
            )
            completed = self._run_job(fast_job)
            self.distributed.call_on_main_and_broadcast(
                lambda: self.selector.record_fast(completed)
            )
            self.distributed.call_on_main_and_broadcast(
                lambda: self.queue.update_event(
                    event_id, fast_job_id=str(fast_job["job_id"])
                )
            )
        if event["epoch_due"]:
            full_candidates = self.distributed.call_on_main_and_broadcast(
                lambda: self._prepare_full_candidates(event_id, snapshot)
            )
            for candidate in full_candidates:
                self._run_job(
                    self.distributed.call_on_main_and_broadcast(
                        lambda candidate=candidate: self.queue.job(
                            candidate["job_id"]
                        )
                    )
                )
            self.distributed.call_on_main_and_broadcast(
                lambda: self._select_epoch(
                    event_id,
                    step=int(event["step"]),
                    epoch=event.get("epoch"),
                )
            )
        self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.update_event(event_id, status="completed")
        )

    def _validate_completed_full_event(
        self, event: Mapping[str, Any]
    ) -> bool:
        reason = self._full_reason(event)
        if reason is None or event.get("status") != "completed":
            raise GRPOValidationError(
                "Train-end reuse requires a completed full-validation event"
            )
        candidates = event.get("full_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GRPOValidationError(
                "Completed full-validation event has no candidates"
            )
        if not any(
            reason in value.get("roles", [])
            for value in candidates
            if isinstance(value, Mapping)
        ):
            raise GRPOValidationError(
                "Completed full-validation event lost its lifecycle role"
            )
        for value in candidates:
            if not isinstance(value, Mapping):
                raise GRPOValidationError(
                    "Completed full-validation candidate is invalid"
                )
            self._validate_completed_job(
                self.queue.job(str(value.get("job_id", "")))
            )
        history = self.selector.read().get("epoch_history")
        if not isinstance(history, list) or not any(
            row.get("event_id") == event["event_id"]
            for row in history
            if isinstance(row, Mapping)
        ):
            raise GRPOValidationError(
                "Completed full-validation event is missing selector history"
            )
        self._validate_selector_state()
        return True

    def _run_job(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        if job["status"] == "completed":
            return self.distributed.call_on_main_and_broadcast(
                lambda: self._validate_completed_job(job)
            )
        running = self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.mark_running(str(job["job_id"]))
        )
        result = self._evaluate(running)
        return self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.mark_completed(str(job["job_id"]), result)
        )

    def _validate_completed_job(
        self,
        job: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if job.get("status") != "completed":
            raise GRPOValidationError(
                "Validation result does not reference a completed queue job"
            )
        mode = str(job.get("mode", ""))
        step = int(job.get("step", -1))
        if mode not in {"fast", "full"} or step < 0:
            raise GRPOValidationError("Completed validation job identity is invalid")
        expected_output = _validated_evaluation_output_path(
            self.validation_dir,
            recorded_output=str(job.get("output_path", "")),
            mode=mode,
            step=step,
        )
        snapshot_record = job.get("snapshot")
        if not isinstance(snapshot_record, Mapping):
            raise GRPOValidationError("Completed validation job lost its snapshot")
        snapshot = self.snapshots.validate(
            str(snapshot_record.get("path", "")),
            expected_step=step,
        )
        if canonical_json(snapshot.as_dict()) != canonical_json(dict(snapshot_record)):
            raise GRPOValidationError("Completed validation snapshot identity changed")
        instr_ids = self.fast_ids if mode == "fast" else self.full_ids
        manifest_path = expected_output / "manifest.json"
        if not manifest_path.is_file():
            raise GRPOValidationError("Completed validation manifest is missing")
        try:
            stored_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GRPOValidationError(
                "Completed validation manifest is invalid"
            ) from exc
        if not isinstance(stored_manifest, Mapping):
            raise GRPOValidationError(
                "Completed validation manifest is not an object"
            )
        if stored_manifest.get("evaluator_family") != NATIVE_EVALUATOR_FAMILY:
            return self._validate_legacy_completed_job(
                job,
                expected_output=expected_output,
                expected_instr_ids=instr_ids,
                snapshot_fingerprint=snapshot.fingerprint,
                adapter_weights_sha256=snapshot.weights_sha256,
            )
        return validate_completed_native_job_output(
            job,
            expected_instr_ids=instr_ids,
            expected_run_fingerprint=self.run_fingerprint,
            expected_validation_fingerprint=str(
                self.contract["validation_fingerprint"]
            ),
            expected_protocol_fingerprint=str(
                self.protocol["protocol_fingerprint"]
            ),
            expected_snapshot_fingerprint=snapshot.fingerprint,
            expected_adapter_weights_sha256=snapshot.weights_sha256,
        )

    def _validate_legacy_completed_job(
        self,
        job: Mapping[str, Any],
        *,
        expected_output: Path,
        expected_instr_ids: Sequence[str],
        snapshot_fingerprint: str,
        adapter_weights_sha256: str,
    ) -> Dict[str, Any]:
        """Semantically revalidate pre-native-manifest internal results."""

        legacy_base = {
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.contract["validation_fingerprint"],
            "job_id": str(job["job_id"]),
            "mode": str(job["mode"]),
            "step": int(job["step"]),
            "snapshot_fingerprint": str(snapshot_fingerprint),
            "adapter_weights_sha256": str(adapter_weights_sha256),
        }
        expected_manifest = build_resumable_evaluation_manifest(
            legacy_base,
            expected_instr_ids=expected_instr_ids,
            world_size=self.distributed.world_size,
        )
        actual_manifest = json.loads(
            (expected_output / "manifest.json").read_text(encoding="utf-8")
        )
        if canonical_json(actual_manifest) != canonical_json(expected_manifest):
            raise GRPOValidationError(
                "Historical completed evaluation manifest changed"
            )
        try:
            predictions = json.loads(
                (expected_output / "predictions.json").read_text(encoding="utf-8")
            )
            per_item = json.loads(
                (expected_output / "per_item_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            metrics = json.loads(
                (expected_output / "metrics.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GRPOValidationError(
                "Historical completed evaluation artifacts are invalid"
            ) from exc
        ids = tuple(str(value) for value in expected_instr_ids)
        if (
            not isinstance(predictions, list)
            or [str(row.get("instr_id", "")) for row in predictions] != list(ids)
            or any(
                row.get("evaluation_fingerprint")
                != expected_manifest["evaluation_fingerprint"]
                or row.get("rank") != index % self.distributed.world_size
                for index, row in enumerate(predictions)
            )
        ):
            raise GRPOValidationError(
                "Historical completed evaluation coverage changed"
            )
        score = self.service.evaluator.evaluate(
            predictions,
            expected_instr_ids=ids,
        )
        expected_result = {
            "evaluation_fingerprint": expected_manifest[
                "evaluation_fingerprint"
            ],
            "count": score["count"],
            "metrics": score["metrics"],
        }
        if (
            canonical_json(metrics) != canonical_json(expected_result)
            or canonical_json(per_item) != canonical_json(score["per_item"])
            or canonical_json(job.get("result"))
            != canonical_json(expected_result)
        ):
            raise GRPOValidationError(
                "Historical completed evaluation metrics changed"
            )
        return dict(job)

    def _validate_selector_state(self) -> bool:
        state = self.selector.read()
        for name, mode in (("quick_best", "fast"), ("full_best", "full")):
            candidate = state.get(name)
            if candidate is None:
                continue
            if not isinstance(candidate, Mapping):
                raise GRPOValidationError(f"Selector {name} is invalid")
            job = self.queue.job(str(candidate.get("job_id", "")))
            if job.get("mode") != mode:
                raise GRPOValidationError(f"Selector {name} references wrong mode")
            validated = self._validate_completed_job(job)
            snapshot = validated["snapshot"]
            result = validated["result"]
            expected = {
                "job_id": str(validated["job_id"]),
                "step": int(validated["step"]),
                "adapter_path": str(snapshot["path"]),
                "snapshot_fingerprint": str(snapshot["fingerprint"]),
                "evaluation_path": str(validated["output_path"]),
                "metrics": dict(result["metrics"]),
            }
            mismatches = {
                key: {"actual": candidate.get(key), "expected": value}
                for key, value in expected.items()
                if canonical_json(candidate.get(key)) != canonical_json(value)
            }
            if mismatches:
                raise GRPOValidationError(
                    f"Selector {name} disagrees with its completed job: "
                    f"{mismatches}"
                )
        return True

    def _evaluate(
        self,
        job: Mapping[str, Any],
    ) -> Dict[str, Any]:
        mode = str(job["mode"])
        step = int(job["step"])
        if (
            job.get("status") not in {"queued", "running"}
            or mode not in {"fast", "full"}
            or step < 0
            or str(job.get("job_id", "")) != f"{mode}-step-{step}"
        ):
            raise GRPOValidationError("Pending validation job identity changed")
        expected_output = _validated_evaluation_output_path(
            self.validation_dir,
            recorded_output=str(job.get("output_path", "")),
            mode=mode,
            step=step,
        )
        instr_ids = self.fast_ids if mode == "fast" else self.full_ids
        snapshot_record = job.get("snapshot")
        if not isinstance(snapshot_record, Mapping):
            raise GRPOValidationError("Pending validation job lost its snapshot")
        snapshot = self.snapshots.validate(
            str(snapshot_record.get("path", "")), expected_step=step
        )
        if canonical_json(snapshot.as_dict()) != canonical_json(
            dict(snapshot_record)
        ):
            raise GRPOValidationError("Pending validation snapshot identity changed")
        policy_identity = self.distributed.call_on_main_and_broadcast(
            lambda: build_native_policy_identity(adapter_path=snapshot.path)
        )
        manifest = {
            "schema_version": NATIVE_EVALUATOR_SCHEMA_VERSION,
            "evaluator_family": NATIVE_EVALUATOR_FAMILY,
            "official_rl_comparable": True,
            "protocol_fingerprint": self.protocol["protocol_fingerprint"],
            "policy_fingerprint": policy_identity["policy_fingerprint"],
            "protocol": self.protocol,
            "policy": policy_identity,
            "candidate_source": "training_validation_snapshot",
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.contract["validation_fingerprint"],
            "job_id": str(job["job_id"]),
            "mode": mode,
            "step": step,
            "snapshot_fingerprint": snapshot.fingerprint,
            "adapter_weights_sha256": snapshot.weights_sha256,
        }
        output = expected_output
        self.distributed.call_on_main_and_broadcast(
            lambda: self._quarantine_incompatible_partial_output(
                job,
                output=output,
                manifest=manifest,
                expected_instr_ids=instr_ids,
            )
        )
        self.distributed.call_on_main_and_broadcast(
            lambda: _initialize_store(
                output, manifest, instr_ids, self.distributed.world_size
            )
        )
        store = ResumableEvaluationStore(
            str(output),
            manifest=manifest,
            expected_instr_ids=instr_ids,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
        )
        with frozen_policy_evaluation(
            self.policy,
            snapshot.path,
        ):
            local = self.service.evaluate_shard(
                self.policy.model,
                self.policy.tokenizer,
                store,
                progress_interval=self.config.progress_interval,
            )
        summaries = self.distributed.all_gather_object(local)
        if not all(row["complete"] for row in summaries):
            raise GRPOValidationError(f"Incomplete validation ranks: {summaries}")
        self.distributed.barrier()
        result = self.distributed.call_on_main_and_broadcast(
            lambda: self.service.finalize(store)
        )
        self.distributed.barrier()
        return result

    def _quarantine_incompatible_partial_output(
        self,
        job: Mapping[str, Any],
        *,
        output: Path,
        manifest: Mapping[str, Any],
        expected_instr_ids: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """Preserve, then replace, a partial result from an older protocol.

        A pre-native rank journal cannot be mixed with rows produced after the
        protocol repair.  Moving the entire old directory keeps every byte for
        audit while allowing the immutable queue job to restart at its
        original canonical output path.
        """

        if not output.exists():
            return None
        if output.is_dir() and not any(output.iterdir()):
            return None
        expected = build_resumable_evaluation_manifest(
            manifest,
            expected_instr_ids=expected_instr_ids,
            world_size=self.distributed.world_size,
        )
        manifest_path = output / "manifest.json" if output.is_dir() else None
        if manifest_path is not None and manifest_path.is_file():
            try:
                observed = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                observed = None
            if isinstance(observed, Mapping) and canonical_json(
                observed
            ) == canonical_json(expected):
                return None

        identity = _recovery_path_identity(output)
        digest = sha256_text(
            canonical_json(
                {
                    "old_output_identity": identity,
                    "expected_evaluation_fingerprint": expected[
                        "evaluation_fingerprint"
                    ],
                }
            )
        )
        mode = str(job["mode"])
        step = int(job["step"])
        recovery_root = (
            self.validation_dir
            / "recovery_quarantine"
            / f"{mode}-step-{step}"
        )
        _require_no_symlink_components(self.validation_dir, recovery_root)
        recovery_root.mkdir(parents=True, exist_ok=True)
        _require_no_symlink_components(self.validation_dir, recovery_root)
        quarantined = recovery_root / digest
        report_path = recovery_root / f"{digest}.migration.json"
        report = {
            "schema_version": 1,
            "action": "preserve_incompatible_partial_and_restart",
            "reason": (
                "partial evaluation manifest is missing or incompatible with "
                "the active native protocol"
            ),
            "job_id": str(job["job_id"]),
            "mode": mode,
            "step": step,
            "original_output_path": str(output),
            "quarantined_output_path": str(quarantined),
            "old_output_identity": identity,
            "expected_evaluation_fingerprint": expected[
                "evaluation_fingerprint"
            ],
            "active_protocol_fingerprint": manifest.get(
                "protocol_fingerprint"
            ),
        }
        _write_recovery_record_once(report_path, report)
        if quarantined.exists():
            raise GRPOValidationError(
                "Recovery quarantine already contains this partial output: "
                f"{quarantined}"
            )
        try:
            output.rename(quarantined)
        except OSError as exc:
            raise GRPOValidationError(
                f"Could not preserve incompatible partial output {output}"
            ) from exc
        return report

    def _initialize_artifacts(self) -> Dict[str, Any]:
        return {
            "queue": self.queue.initialize(),
            "selector": self.selector.initialize(),
        }

    def _snapshot_event(self, event_id: str) -> Dict[str, Any]:
        event = next(
            row
            for row in self.queue.read()["events"]
            if row["event_id"] == event_id
        )
        if event["snapshot"] is None:
            snapshot = self.snapshots.create(
                str(event["source_path"]), step=int(event["step"])
            )
            event = self.queue.update_event(event_id, snapshot=snapshot.as_dict())
        else:
            value = event["snapshot"]
            snapshot = self.snapshots.validate(
                str(value["path"]), expected_step=int(event["step"])
            )
            if canonical_json(snapshot.as_dict()) != canonical_json(value):
                raise GRPOValidationError("Queued eval snapshot identity changed")
        return dict(event["snapshot"])

    def _enqueue_job(
        self, snapshot: Mapping[str, Any], mode: str
    ) -> Dict[str, Any]:
        step = int(snapshot["step"])
        job_id = f"{mode}-step-{step}"
        return self.queue.enqueue_job(
            {
                "job_id": job_id,
                "mode": mode,
                "step": step,
                "snapshot": dict(snapshot),
                "output_path": str(
                    self.validation_dir / "evaluations" / mode / f"step-{step}"
                ),
            }
        )

    def _prepare_full_candidates(
        self, event_id: str, current_snapshot: Mapping[str, Any]
    ) -> Sequence[Dict[str, Any]]:
        self._validate_selector_state()
        event = next(
            row
            for row in self.queue.read()["events"]
            if row["event_id"] == event_id
        )
        if event["full_candidates"]:
            return tuple(event["full_candidates"])
        full_reason = self._full_reason(event)
        if full_reason is None:
            raise GRPOValidationError(
                "Full candidates requested for a fast-only event"
            )
        grouped: Dict[str, Dict[str, Any]] = {
            str(current_snapshot["fingerprint"]): {
                "snapshot": dict(current_snapshot),
                "roles": [full_reason],
            }
        }
        quick = self.selector.read().get("quick_best")
        if quick is not None:
            fingerprint = str(quick["snapshot_fingerprint"])
            if fingerprint in grouped:
                grouped[fingerprint]["roles"].append("quick_best")
            else:
                snapshot = self.snapshots.validate(
                    str(quick["adapter_path"]), expected_step=int(quick["step"])
                )
                grouped[fingerprint] = {
                    "snapshot": snapshot.as_dict(),
                    "roles": ["quick_best"],
                }
        candidates = []
        for value in grouped.values():
            job = self._enqueue_job(value["snapshot"], "full")
            candidates.append(
                {"job_id": str(job["job_id"]), "roles": list(value["roles"])}
            )
        self.queue.update_event(event_id, full_candidates=candidates)
        return tuple(candidates)

    def _select_epoch(
        self, event_id: str, *, step: int, epoch: Any
    ) -> Dict[str, Any]:
        self._validate_selector_state()
        event = next(
            row
            for row in self.queue.read()["events"]
            if row["event_id"] == event_id
        )
        candidates = [
            completed_candidate(
                self.queue.job(str(value["job_id"])), roles=value["roles"]
            )
            for value in event["full_candidates"]
        ]
        return self.selector.record_epoch(
            event_id=event_id,
            step=step,
            epoch=epoch,
            candidates=candidates,
        )


def make_grpo_validation_callback(
    manager: GRPOValidationManager,
    *,
    transformers_module: Any,
) -> Any:
    base = getattr(transformers_module, "TrainerCallback", object)

    class ValidationCallback(base):
        def __init__(self) -> None:
            self.epoch_steps: Dict[int, float] = {}
            self.saved_steps = set()

        @staticmethod
        def _max_steps(args: Any) -> int:
            try:
                return int(getattr(args, "max_steps", -1))
            except (TypeError, ValueError):
                return -1

        @staticmethod
        def _epoch(state: Any) -> Optional[float]:
            value = getattr(state, "epoch", None)
            return None if value is None else float(value)

        @staticmethod
        def _checkpoint(args: Any, step: int) -> str:
            return str(
                Path(args.output_dir).resolve() / f"checkpoint-{int(step)}"
            )

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            step = int(state.global_step)
            max_steps = self._max_steps(args)
            fast_due = bool(
                step and step % manager.config.fast_interval_steps == 0
            )
            # on_train_end cannot ask Trainer to materialize a checkpoint.
            # Force the terminal optimizer step through the normal save path,
            # where the checkpoint callback writes and audits all resume files.
            max_step_due = bool(max_steps > 0 and step >= max_steps)
            if fast_due or max_step_due:
                control.should_save = True
            return control

        def on_epoch_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            step = int(state.global_step)
            epoch = float(state.epoch)
            if step in self.saved_steps:
                manager.run_scheduled_checkpoint(
                    step=step,
                    checkpoint_path=self._checkpoint(args, step),
                    fast_due=False,
                    epoch_due=True,
                    epoch=epoch,
                )
                control.should_save = False
            else:
                self.epoch_steps[step] = epoch
                control.should_save = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            step = int(state.global_step)
            self.saved_steps.add(step)
            epoch = self.epoch_steps.pop(step, None)
            manager.run_scheduled_checkpoint(
                step=step,
                checkpoint_path=self._checkpoint(args, step),
                fast_due=bool(step and step % manager.config.fast_interval_steps == 0),
                epoch_due=epoch is not None,
                epoch=epoch,
            )
            return control

        def on_train_end(
            self, args: Any, state: Any, control: Any, **_: Any
        ) -> Any:
            step = int(state.global_step)
            max_steps = self._max_steps(args)
            if max_steps <= 0 or step < max_steps:
                return control
            manager.ensure_train_end_validation(
                step=step,
                checkpoint_path=self._checkpoint(args, step),
                epoch=self._epoch(state),
            )
            return control

    return ValidationCallback()


@contextmanager
def frozen_policy_evaluation(
    policy: Any,
    adapter_path: Optional[str],
) -> Iterator[None]:
    import numpy as np
    import torch

    model = policy.model
    parameters = list(model.named_parameters())
    flags = {name: parameter.requires_grad for name, parameter in parameters}
    training = model.training
    use_cache = getattr(model.config, "use_cache", None)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    saved_lora = None
    try:
        if adapter_path is not None:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            saved_lora = {
                name: parameter.detach().cpu().clone()
                for name, parameter in parameters
                if ".lora_A." in f".{name}." or ".lora_B." in f".{name}."
            }
            set_peft_model_state_dict(
                model,
                load_file(str(Path(adapter_path) / "adapter_model.safetensors")),
                adapter_name="default",
            )
        for _, parameter in parameters:
            parameter.requires_grad_(False)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        yield
    finally:
        if saved_lora is not None:
            with torch.no_grad():
                for name, parameter in parameters:
                    if name in saved_lora:
                        parameter.copy_(saved_lora[name].to(parameter.device))
        for name, parameter in parameters:
            parameter.requires_grad_(flags[name])
        model.train(training)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)


def validate_completed_native_job_output(
    job: Mapping[str, Any],
    *,
    expected_instr_ids: Sequence[str],
    expected_run_fingerprint: str,
    expected_validation_fingerprint: str,
    expected_protocol_fingerprint: str,
    expected_snapshot_fingerprint: str,
    expected_adapter_weights_sha256: str,
) -> Dict[str, Any]:
    """Revalidate a completed queue job before any selector consumes it."""

    if job.get("status") != "completed":
        raise GRPOValidationError("Validation job is not completed")
    result = job.get("result")
    if not isinstance(result, Mapping):
        raise GRPOValidationError("Completed validation job has no result")
    output = Path(str(job.get("output_path", ""))).expanduser().resolve()
    manifest = load_official_native_manifest(str(output))
    expected_ids = tuple(str(value) for value in expected_instr_ids)
    checks = {
        "run_fingerprint": str(expected_run_fingerprint),
        "validation_fingerprint": str(expected_validation_fingerprint),
        "job_id": str(job.get("job_id", "")),
        "mode": str(job.get("mode", "")),
        "step": int(job.get("step", -1)),
        "snapshot_fingerprint": str(expected_snapshot_fingerprint),
        "adapter_weights_sha256": str(expected_adapter_weights_sha256),
        "protocol_fingerprint": str(expected_protocol_fingerprint),
        "expected_instr_id_count": len(expected_ids),
        "expected_instr_ids_sha256": sha256_text(
            canonical_json(list(expected_ids))
        ),
    }
    mismatches = {
        name: {"actual": manifest.get(name), "expected": expected}
        for name, expected in checks.items()
        if manifest.get(name) != expected
    }
    if mismatches:
        raise GRPOValidationError(
            f"Completed native evaluation provenance changed: {mismatches}"
        )
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if canonical_json(metrics) != canonical_json(dict(result)):
        raise GRPOValidationError(
            "Completed native evaluation metrics disagree with the queue"
        )
    return dict(job)


def _recovery_path_identity(path: Path) -> Dict[str, Any]:
    """Inventory every preserved byte before quarantining a stale output."""

    path = _lexical_absolute_path(path)
    if path.is_symlink():
        raise GRPOValidationError(
            f"Refusing to quarantine a symlinked evaluation output: {path}"
        )
    if path.is_file():
        entries = [
            {
                "path": path.name,
                "kind": "file",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        ]
        root_kind = "file"
    elif path.is_dir():
        entries = []
        for candidate in sorted(path.rglob("*")):
            if candidate.is_symlink():
                raise GRPOValidationError(
                    "Refusing to quarantine a partial output containing a "
                    f"symlink: {candidate}"
                )
            relative = candidate.relative_to(path).as_posix()
            if candidate.is_dir():
                entries.append({"path": relative, "kind": "directory"})
            elif candidate.is_file():
                entries.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "size_bytes": candidate.stat().st_size,
                        "sha256": sha256_file(candidate),
                    }
                )
            else:
                raise GRPOValidationError(
                    "Partial evaluation output contains an unsupported entry: "
                    f"{candidate}"
                )
        root_kind = "directory"
    else:
        raise GRPOValidationError(
            f"Partial evaluation output has an unsupported type: {path}"
        )
    body = {
        "schema_version": 1,
        "path": str(path),
        "root_kind": root_kind,
        "entries": entries,
    }
    body["identity_sha256"] = sha256_text(canonical_json(body))
    return body


def _lexical_absolute_path(path: Any) -> Path:
    """Normalize ``..`` and ``~`` without following filesystem symlinks."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _validated_evaluation_output_path(
    validation_dir: Path,
    *,
    recorded_output: str,
    mode: str,
    step: int,
) -> Path:
    """Bind a queue output to its canonical location and reject symlinks."""

    root = _lexical_absolute_path(validation_dir)
    expected = _lexical_absolute_path(
        root / "evaluations" / mode / f"step-{step}"
    )
    actual = _lexical_absolute_path(recorded_output)
    if actual != expected:
        raise GRPOValidationError("Validation output path changed")
    _require_no_symlink_components(root, expected)
    return expected


def _require_no_symlink_components(root: Path, path: Path) -> None:
    """Reject filesystem indirection below one already trusted run root."""

    root = _lexical_absolute_path(root)
    path = _lexical_absolute_path(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise GRPOValidationError(
            f"Validation path escapes its run root: {path}"
        ) from exc
    cursor = root
    if cursor.is_symlink():
        raise GRPOValidationError(
            f"Validation directory must not be a symlink: {cursor}"
        )
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GRPOValidationError(
                f"Validation output path contains a symlink: {cursor}"
            )


def _write_recovery_record_once(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create an immutable recovery record, or verify it exists."""

    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GRPOValidationError(
                f"Recovery record is invalid: {path}"
            ) from exc
        if canonical_json(observed) != canonical_json(dict(value)):
            raise GRPOValidationError(f"Recovery record changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(observed) != canonical_json(dict(value)):
                raise GRPOValidationError(f"Recovery record changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _initialize_store(
    output: Path,
    manifest: Mapping[str, Any],
    instr_ids: Sequence[str],
    world_size: int,
) -> bool:
    ResumableEvaluationStore(
        str(output),
        manifest=manifest,
        expected_instr_ids=instr_ids,
        rank=0,
        world_size=world_size,
    )
    return True
