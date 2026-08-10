"""Minimal in-training Val-Unseen scheduling and best-LoRA selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from action_plan_cache import canonical_json, sha256_file, sha256_text
from grpo_eval_artifacts import (
    BestSelector,
    EvaluationQueue,
    EvaluationSnapshotStore,
    completed_candidate,
)
from r2r_evaluation import (
    R2REvaluationConfig,
    ResumableEvaluationStore,
    StandardR2REvaluator,
    build_validation_environment_factory,
    evaluate_policy_shard,
    load_fast_subset_manifest,
    load_validation_dataset,
    prepare_fast_subset_manifest,
)


class GRPOValidationError(RuntimeError):
    pass


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
        self.factory = build_validation_environment_factory(dataset)
        self.evaluator = StandardR2REvaluator(
            dataset.records,
            dataset.config.connectivity_dir,
            graph_cache=self.factory.graph_cache,
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

    def resume_pending(self, *, current_step: int) -> int:
        pending = self.distributed.call_on_main_and_broadcast(
            lambda: list(self.queue.pending_events())
        )
        for event in pending:
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
    ) -> None:
        if not fast_due and not epoch_due:
            return
        event = {
            "event_id": (
                f"step-{int(step)}-fast-{int(bool(fast_due))}"
                f"-epoch-{int(bool(epoch_due))}"
            ),
            "step": int(step),
            "source_path": str(Path(checkpoint_path).resolve()),
            "fast_due": bool(fast_due),
            "epoch_due": bool(epoch_due),
            "epoch": epoch,
        }
        queued = self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.enqueue_event(event)
        )
        if queued["status"] != "completed":
            self._execute_event(queued)

    def _execute_event(self, event: Mapping[str, Any]) -> None:
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

    def _run_job(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        if job["status"] == "completed":
            return dict(job)
        running = self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.mark_running(str(job["job_id"]))
        )
        result = self._evaluate(running)
        return self.distributed.call_on_main_and_broadcast(
            lambda: self.queue.mark_completed(str(job["job_id"]), result)
        )

    def _evaluate(
        self,
        job: Mapping[str, Any],
    ) -> Dict[str, Any]:
        mode = str(job["mode"])
        step = int(job["step"])
        instr_ids = self.fast_ids if mode == "fast" else self.full_ids
        snapshot = self.snapshots.validate(
            str(job["snapshot"]["path"]), expected_step=step
        )
        manifest = {
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.contract["validation_fingerprint"],
            "job_id": str(job["job_id"]),
            "mode": mode,
            "step": step,
            "snapshot_fingerprint": snapshot.fingerprint,
            "adapter_weights_sha256": snapshot.weights_sha256,
        }
        output = Path(str(job["output_path"]))
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
            local = evaluate_policy_shard(
                self.policy.model,
                self.policy.tokenizer,
                self.dataset,
                store,
                environment_factory=self.factory,
                progress_interval=self.config.progress_interval,
            )
        summaries = self.distributed.all_gather_object(local)
        if not all(row["complete"] for row in summaries):
            raise GRPOValidationError(f"Incomplete validation ranks: {summaries}")
        self.distributed.barrier()
        result = self.distributed.call_on_main_and_broadcast(
            lambda: store.finalize(self.evaluator)
        )
        self.distributed.barrier()
        return result

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
        event = next(
            row
            for row in self.queue.read()["events"]
            if row["event_id"] == event_id
        )
        if event["full_candidates"]:
            return tuple(event["full_candidates"])
        grouped: Dict[str, Dict[str, Any]] = {
            str(current_snapshot["fingerprint"]): {
                "snapshot": dict(current_snapshot),
                "roles": ["epoch_end"],
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

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            step = int(state.global_step)
            if step and step % manager.config.fast_interval_steps == 0:
                control.should_save = True
            return control

        def on_epoch_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            step = int(state.global_step)
            epoch = float(state.epoch)
            if step in self.saved_steps:
                manager.run_scheduled_checkpoint(
                    step=step,
                    checkpoint_path=str(
                        Path(args.output_dir).resolve() / f"checkpoint-{step}"
                    ),
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
                checkpoint_path=str(
                    Path(args.output_dir).resolve() / f"checkpoint-{step}"
                ),
                fast_due=bool(step and step % manager.config.fast_interval_steps == 0),
                epoch_due=epoch is not None,
                epoch=epoch,
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
