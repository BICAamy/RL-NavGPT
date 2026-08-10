"""Minimal in-training Val-Unseen scheduling and best-LoRA selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from action_plan_cache import canonical_json, sha256_file, sha256_text
from lora_policy import ADAPTER_MANIFEST_NAME, validate_local_adapter_directory
from r2r_evaluation import (
    R2REvaluationConfig,
    ResumableEvaluationStore,
    StandardR2REvaluator,
    build_validation_environment_factory,
    evaluate_policy_shard,
    load_fast_subset_manifest,
    load_validation_dataset,
    prepare_fast_subset_manifest,
    selection_key,
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
        self.distributed.call_on_main_and_broadcast(self._initialize_state)

    def resume_pending(self, *, current_step: int) -> None:
        state = self._state()
        pending = state.get("pending")
        if pending is None:
            return
        if int(pending["step"]) != int(current_step):
            raise GRPOValidationError(
                f"Pending validation is step {pending['step']}, not {current_step}"
            )
        self._execute(pending)

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
        workflow = {
            "step": int(step),
            "candidate_path": str(Path(checkpoint_path).resolve()),
            "fast_due": bool(fast_due),
            "epoch_due": bool(epoch_due),
            "epoch": epoch,
        }
        pending = self.distributed.call_on_main_and_broadcast(
            lambda: self._begin(workflow)
        )
        if not pending.get("complete"):
            self._execute(pending)

    def _execute(self, workflow: Mapping[str, Any]) -> None:
        step = int(workflow["step"])
        path = str(workflow["candidate_path"])
        if workflow["fast_due"]:
            result = self._evaluate(step, path, "fast", self.fast_ids, current=True)
            self.distributed.call_on_main_and_broadcast(
                lambda: self._record_fast(step, path, result["metrics"])
            )
        if workflow["epoch_due"]:
            current = self._evaluate(step, path, "full", self.full_ids, current=True)
            state = self._state()
            quick = state.get("quick_best")
            quick_metrics = None
            if quick is not None:
                if int(quick["step"]) == step:
                    quick_metrics = current["metrics"]
                else:
                    quick_metrics = self._evaluate(
                        int(quick["step"]),
                        str(quick["adapter_path"]),
                        "full",
                        self.full_ids,
                        current=False,
                    )["metrics"]
            self.distributed.call_on_main_and_broadcast(
                lambda: self._record_epoch(
                    step=step,
                    epoch=workflow.get("epoch"),
                    source_path=path,
                    metrics=current["metrics"],
                    quick=quick,
                    quick_metrics=quick_metrics,
                )
            )
        self.distributed.call_on_main_and_broadcast(
            lambda: self._finish(workflow)
        )

    def _evaluate(
        self,
        step: int,
        adapter_path: str,
        mode: str,
        instr_ids: Sequence[str],
        *,
        current: bool,
    ) -> Dict[str, Any]:
        adapter = validate_local_adapter_directory(adapter_path, self.policy.config)
        manifest = {
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.contract["validation_fingerprint"],
            "mode": mode,
            "step": step,
            "adapter_weights_sha256": sha256_file(
                adapter / "adapter_model.safetensors"
            ),
        }
        output = self.validation_dir / "evaluations" / mode / f"step-{step}"
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
            None if current else str(adapter),
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

    def _record_fast(
        self, step: int, source_path: str, metrics: Mapping[str, Any]
    ) -> Dict[str, Any]:
        state = self._read_state()
        if any(int(row["step"]) == step for row in state["fast_history"]):
            return state
        previous = state.get("quick_best")
        improved = previous is None or selection_key(metrics, step=step) > selection_key(
            previous["metrics"], step=int(previous["step"])
        )
        pinned = self._pin(source_path, step) if improved else None
        if improved:
            state["quick_best"] = {
                "step": step,
                "adapter_path": pinned,
                "metrics": dict(metrics),
            }
        state["fast_history"].append(
            {"step": step, "metrics": dict(metrics), "improved": improved}
        )
        self._write_state(state)
        return state

    def _record_epoch(
        self,
        *,
        step: int,
        epoch: Any,
        source_path: str,
        metrics: Mapping[str, Any],
        quick: Optional[Mapping[str, Any]],
        quick_metrics: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        state = self._read_state()
        if any(int(row["step"]) == step for row in state["epoch_history"]):
            return state
        candidates = [
            {"role": "epoch_end", "step": step, "path": source_path, "metrics": dict(metrics)}
        ]
        if quick is not None and quick_metrics is not None:
            candidates.append(
                {
                    "role": "quick_best",
                    "step": int(quick["step"]),
                    "path": str(quick["adapter_path"]),
                    "metrics": dict(quick_metrics),
                }
            )
        if state.get("full_best") is not None:
            old = state["full_best"]
            candidates.append(
                {
                    "role": "previous_full_best",
                    "step": int(old["step"]),
                    "path": str(old["adapter_path"]),
                    "metrics": dict(old["metrics"]),
                }
            )
        winner = max(
            candidates,
            key=lambda row: selection_key(row["metrics"], step=int(row["step"])),
        )
        state["full_best"] = {
            "step": int(winner["step"]),
            "adapter_path": self._pin(winner["path"], int(winner["step"])),
            "metrics": winner["metrics"],
            "selected_role": winner["role"],
        }
        state["epoch_history"].append(
            {"step": step, "epoch": epoch, "winner": state["full_best"]}
        )
        self._write_state(state)
        return state

    def _pin(self, source_path: str, step: int) -> str:
        source = validate_local_adapter_directory(source_path, self.policy.config)
        destination = self.validation_dir / "pinned" / f"step-{step}"
        if destination.exists():
            validate_local_adapter_directory(str(destination), self.policy.config)
            if sha256_file(destination / "adapter_model.safetensors") != sha256_file(
                source / "adapter_model.safetensors"
            ):
                raise GRPOValidationError(f"Pinned adapter differs: {destination}")
            return str(destination)
        destination.mkdir(parents=True)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            ADAPTER_MANIFEST_NAME,
        ):
            shutil.copy2(source / name, destination / name)
        return str(destination)

    def _initialize_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = self._read_state()
            self._check_state(state)
            return state
        state = {
            "schema_version": 1,
            "run_fingerprint": self.run_fingerprint,
            "validation_fingerprint": self.contract["validation_fingerprint"],
            "pending": None,
            "quick_best": None,
            "full_best": None,
            "fast_history": [],
            "epoch_history": [],
            "completed_workflows": [],
        }
        self._write_state(state)
        return state

    def _begin(self, workflow: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._read_state()
        if state["pending"] is not None:
            if canonical_json(state["pending"]) != canonical_json(workflow):
                raise GRPOValidationError("Another validation workflow is pending")
            return state["pending"]
        if any(
            int(row["step"]) == int(workflow["step"])
            and row["fast_due"] == workflow["fast_due"]
            and row["epoch_due"] == workflow["epoch_due"]
            for row in state["completed_workflows"]
        ):
            return {**dict(workflow), "complete": True}
        state["pending"] = dict(workflow)
        self._write_state(state)
        return dict(workflow)

    def _finish(self, workflow: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._read_state()
        state["completed_workflows"].append(dict(workflow))
        state["pending"] = None
        self._write_state(state)
        return state

    def _state(self) -> Dict[str, Any]:
        return self.distributed.call_on_main_and_broadcast(self._read_state)

    def _read_state(self) -> Dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _check_state(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("run_fingerprint") != self.run_fingerprint
            or state.get("validation_fingerprint")
            != self.contract["validation_fingerprint"]
        ):
            raise GRPOValidationError("Validation state belongs to another run")


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
                    checkpoint_path=str(Path(args.output_dir).resolve() / f"checkpoint-{step}"),
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
                checkpoint_path=str(Path(args.output_dir).resolve() / f"checkpoint-{step}"),
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
