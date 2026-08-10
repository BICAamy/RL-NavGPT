"""Small dependency-light contract test for training-time R2R validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import networkx as nx


NAV_SRC = Path(__file__).resolve().parents[1]
if str(NAV_SRC) not in sys.path:
    sys.path.insert(0, str(NAV_SRC))

from grpo_eval_artifacts import (  # noqa: E402
    ADAPTER_FILES,
    BestSelector,
    EvaluationArtifactError,
    EvaluationQueue,
    EvaluationSnapshotStore,
    completed_candidate,
)
from grpo_validation import make_grpo_validation_callback  # noqa: E402
from r2r_evaluation import (  # noqa: E402
    ResumableEvaluationStore,
    StandardR2REvaluator,
    prepare_fast_subset_manifest,
    selection_key,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class GraphCache:
    def __init__(self) -> None:
        graph = nx.Graph()
        graph.add_edge("a", "b", weight=1.0)
        graph.add_edge("b", "c", weight=1.0)
        self.graphs = {"scan": graph}
        self.shortest_distances = {
            "scan": dict(nx.all_pairs_dijkstra_path_length(graph))
        }


def rows(count: int = 6):
    return [
        {
            "instr_id": f"route_{index}",
            "instruction": f"instruction {index}",
            "scan": "scan",
            "path_id": index,
            "path": ["a", "b", "c"],
            "heading": 0.0,
        }
        for index in range(count)
    ]


def prediction(instr_id: str):
    return {
        "instr_id": instr_id,
        "scan": "scan",
        "trajectory_path": ["a", "b", "c"],
        "step_count": 2,
        "termination_reason": "success",
        "decisions": [],
    }


class FakeManager:
    def __init__(self) -> None:
        self.config = SimpleNamespace(fast_interval_steps=1_000)
        self.calls = []

    def run_scheduled_checkpoint(self, **kwargs):
        self.calls.append(kwargs)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="navgpt-r2r-contract-") as temp:
        root = Path(temp)
        annotation = root / "val.json"
        annotation.write_text(json.dumps(rows()), encoding="utf-8")
        subset = root / "fast.json"
        first = prepare_fast_subset_manifest(
            str(annotation),
            str(subset),
            subset_size=3,
            seed=0,
            expected_instruction_count=6,
        )
        second = prepare_fast_subset_manifest(
            str(annotation),
            str(subset),
            subset_size=3,
            seed=0,
            expected_instruction_count=6,
        )
        require(first == second, "Fast subset is not fixed")

        evaluator = StandardR2REvaluator(
            rows(), str(root), graph_cache=GraphCache()
        )
        score = evaluator.evaluate(
            [prediction(row["instr_id"]) for row in rows()],
            expected_instr_ids=[row["instr_id"] for row in rows()],
        )
        require(score["metrics"]["spl"] == 100.0, "Oracle SPL changed")
        require(score["metrics"]["sr"] == 100.0, "Oracle SR changed")

        ids = tuple(evaluator.items)
        stores = [
            ResumableEvaluationStore(
                str(root / "evaluation"),
                manifest={"mode": "fast", "candidate": "synthetic"},
                expected_instr_ids=ids,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]
        for store in stores:
            for instr_id in store.assigned_instr_ids:
                store.append(prediction(instr_id))
        with stores[0].journal_path.open("ab") as file_obj:
            file_obj.write(b'{"interrupted":')
        recovered = ResumableEvaluationStore(
            str(root / "evaluation"),
            manifest={"mode": "fast", "candidate": "synthetic"},
            expected_instr_ids=ids,
            rank=0,
            world_size=2,
        )
        require(not recovered.pending_instr_ids(), "JSONL recovery lost rows")
        require(
            recovered.finalize(evaluator)["count"] == len(ids),
            "Final coverage changed",
        )

        adapter = root / "checkpoint-1000"
        adapter.mkdir()
        for name in ADAPTER_FILES:
            (adapter / name).write_text(f"{name}:1000\n", encoding="utf-8")
        snapshots = EvaluationSnapshotStore(
            str(root / "snapshots"),
            policy_config=None,
            run_fingerprint="run",
            validation_fingerprint="validation",
            adapter_validator=lambda value: Path(value).resolve(),
        )
        snapshot = snapshots.create(str(adapter), step=1_000)
        adapter.joinpath("adapter_model.safetensors").write_text(
            "changed\n", encoding="utf-8"
        )
        try:
            snapshots.create(str(adapter), step=1_000)
        except EvaluationArtifactError:
            pass
        else:
            raise AssertionError("Snapshot silently followed a changed checkpoint")
        snapshot_weights = Path(snapshot.path) / "adapter_model.safetensors"
        snapshot_weights.write_text("tampered\n", encoding="utf-8")
        try:
            snapshots.validate(snapshot.path, expected_step=1_000)
        except EvaluationArtifactError:
            pass
        else:
            raise AssertionError("Tampered snapshot passed validation")
        snapshot_weights.write_text(
            "adapter_model.safetensors:1000\n", encoding="utf-8"
        )
        snapshots.validate(snapshot.path, expected_step=1_000)

        queue = EvaluationQueue(
            str(root / "queue.json"),
            run_fingerprint="run",
            validation_fingerprint="validation",
        )
        queue.initialize()
        event = queue.enqueue_event(
            {
                "event_id": "step-1000-fast-1-epoch-0",
                "step": 1_000,
                "source_path": str(adapter),
                "fast_due": True,
                "epoch_due": False,
                "epoch": None,
            }
        )
        queue.update_event(event["event_id"], snapshot=snapshot.as_dict())
        job = queue.enqueue_job(
            {
                "job_id": "fast-step-1000",
                "mode": "fast",
                "step": 1_000,
                "snapshot": snapshot.as_dict(),
                "output_path": str(root / "fast-1000"),
            }
        )
        queue.mark_running(job["job_id"])
        fast_metrics = {"spl": 50, "sr": 60, "nDTW": 70, "nav_error": 4}
        completed_fast = queue.mark_completed(
            job["job_id"], {"count": 128, "metrics": fast_metrics}
        )
        require(
            EvaluationQueue(
                str(root / "queue.json"),
                run_fingerprint="run",
                validation_fingerprint="validation",
            ).job(job["job_id"])["status"]
            == "completed",
            "Evaluation queue did not survive reload",
        )

        selector = BestSelector(
            str(root / "best.json"),
            run_fingerprint="run",
            validation_fingerprint="validation",
        )
        selector.initialize()
        selector.record_fast(completed_fast)
        full_job = {
            **completed_fast,
            "job_id": "full-step-1000",
            "mode": "full",
            "result": {"count": 6, "metrics": fast_metrics},
        }
        full_candidate = completed_candidate(full_job, roles=("epoch_end",))
        selected = selector.record_epoch(
            event_id="epoch-1",
            step=1_000,
            epoch=1.0,
            candidates=[full_candidate],
        )
        require(
            selected["full_best"]["adapter_path"] == snapshot.path,
            "Best selector did not retain the immutable snapshot",
        )
        queue.update_event(event["event_id"], status="completed")
        require(not queue.pending_events(), "Completed event remained queued")

        strong = {"spl": 50, "sr": 60, "nDTW": 70, "nav_error": 4}
        weak = {"spl": 49, "sr": 99, "nDTW": 99, "nav_error": 0}
        require(
            selection_key(strong, step=2_000)
            > selection_key(weak, step=1_000),
            "SPL is not the primary selector",
        )

        manager = FakeManager()
        callback = make_grpo_validation_callback(
            manager,
            transformers_module=SimpleNamespace(TrainerCallback=object),
        )
        args = SimpleNamespace(output_dir=str(root / "run"))
        state = SimpleNamespace(global_step=1_000, epoch=1.0)
        control = SimpleNamespace(should_save=False)
        callback.on_step_end(args, state, control)
        callback.on_epoch_end(args, state, control)
        callback.on_save(args, state, control)
        require(len(manager.calls) == 1, "Combined fast/epoch event duplicated")
        require(
            manager.calls[0]["fast_due"] and manager.calls[0]["epoch_due"],
            "Fast or epoch validation was not scheduled",
        )

    print("PASS R2R validation contract")
    print("- fixed Val-Unseen subset and standard metrics")
    print("- rank JSONL recovery and exact coverage")
    print("- immutable eval snapshot and resumable evaluation queue")
    print("- SPL-first quick/full best selector")
    print("- 1000-step fast plus epoch-end full scheduling")


if __name__ == "__main__":
    main()
