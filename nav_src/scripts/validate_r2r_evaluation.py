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
    print("- 1000-step fast plus epoch-end full scheduling")


if __name__ == "__main__":
    main()
