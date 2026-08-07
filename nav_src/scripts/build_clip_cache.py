"""Build and verify raw-visual and instruction CLIP feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import numpy as np


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import load_annotation_instructions  # noqa: E402
from clip_feature_cache import (  # noqa: E402
    CLIP_CACHE_SCHEMA_VERSION,
    CLIPTextFeatureEncoder,
    InstructionCLIPFeatureStore,
    VISUAL_MANIFEST_NAME,
    VisualCLIPFeatureStore,
    sha256_file,
    sha256_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CLIP caches from original R2R instructions and RGB views"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text")
    _add_model_arguments(text_parser)
    text_parser.add_argument("--annotation", required=True)
    text_parser.add_argument("--output", required=True)
    text_parser.add_argument("--expected-count", type=int, default=0)
    text_parser.add_argument("--batch-size", type=int, default=128)
    text_parser.add_argument("--overwrite", action="store_true")

    visual_parser = subparsers.add_parser("visual")
    _add_model_arguments(visual_parser)
    visual_parser.add_argument("--annotation", required=True)
    visual_parser.add_argument("--connectivity-dir", required=True)
    visual_parser.add_argument("--scan-data-dir", required=True)
    visual_parser.add_argument("--output-dir", required=True)
    visual_parser.add_argument("--batch-size", type=int, default=36)
    visual_parser.add_argument("--width", type=int, default=640)
    visual_parser.add_argument("--height", type=int, default=480)
    visual_parser.add_argument("--vfov", type=float, default=60.0)
    visual_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--annotation", required=True)
    verify_parser.add_argument("--connectivity-dir", required=True)
    verify_parser.add_argument("--instruction-cache", required=True)
    verify_parser.add_argument("--visual-cache-dir", required=True)
    verify_parser.add_argument(
        "--model-id",
        default="openai/clip-vit-large-patch14",
    )
    verify_parser.add_argument("--model-revision", default="main")
    verify_parser.add_argument("--expected-count", type=int, default=0)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-id",
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
    )


def build_text_cache(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    records = load_annotation_instructions(args.annotation)
    _check_expected_count(records, args.expected_count)
    output = Path(args.output)
    manifest_path = Path(f"{output}.manifest.json")
    if (output.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}; pass --overwrite to replace it"
        )
    model_provenance = _model_weight_provenance(Path(args.model_path))
    encoder = CLIPTextFeatureEncoder(
        args.model_path,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        cache_size=max(20_000, len(records)),
        local_files_only=True,
    )
    instructions = [str(record["instruction"]) for record in records]
    features = encoder.encode_many(instructions, batch_size=args.batch_size)
    features = features.astype(np.float16)
    arrays = {
        "instr_ids": np.asarray(
            [str(record["instr_id"]) for record in records]
        ),
        "instruction_sha256": np.asarray(
            [sha256_text(instruction) for instruction in instructions]
        ),
        "features": features,
    }
    _write_npz_atomic(output, arrays)
    manifest = _base_manifest(
        args,
        "instruction",
        features.shape[-1],
        model_provenance,
    )
    manifest.update(
        {
            "record_count": len(records),
            "annotation_sha256": sha256_file(Path(args.annotation)),
            "cache_sha256": sha256_file(output),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    print(
        f"Wrote {len(records)} instruction features with dimension "
        f"{features.shape[-1]} to {output}"
    )


def build_visual_cache(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if not 0.0 < args.vfov < 180.0:
        raise ValueError("--vfov must be between 0 and 180 degrees")
    records = load_annotation_instructions(args.annotation)
    scans = sorted({str(record["scan"]) for record in records})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / VISUAL_MANIFEST_NAME
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Visual manifest already exists: {manifest_path}; "
            "pass --overwrite to rebuild"
        )
    model_provenance = _model_weight_provenance(Path(args.model_path))

    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPVisionModelWithProjection

    device = _resolve_device(args.device, torch)
    if device == "cpu" and args.dtype != "fp32":
        raise ValueError("CPU visual CLIP encoding requires --dtype fp32")
    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    model = CLIPVisionModelWithProjection.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    feature_dim = int(model.config.projection_dim)
    simulator = _build_rendering_simulator(args)
    scan_provenance = _scan_provenance(args, model_provenance)

    scan_records: Dict[str, Dict[str, Any]] = {}
    for scan_index, scan in enumerate(scans, start=1):
        viewpoints = _included_viewpoints(
            Path(args.connectivity_dir) / f"{scan}_connectivity.json"
        )
        path = output_dir / f"{scan}.npz"
        if path.exists() and not args.overwrite:
            scan_records[scan] = _resume_scan_record(
                path,
                viewpoints,
                feature_dim,
                scan_provenance,
            )
            print(
                f"[{scan_index}/{len(scans)}] {scan}: "
                f"resumed {len(viewpoints)} viewpoints"
            )
            continue
        features = np.empty(
            (len(viewpoints), 36, feature_dim),
            dtype=np.float16,
        )
        for viewpoint_index, viewpoint in enumerate(viewpoints):
            images = _render_36_views(
                simulator,
                scan,
                viewpoint,
                Image,
            )
            encoded = _encode_images(
                model,
                processor,
                images,
                device=device,
                dtype=torch_dtype,
                batch_size=args.batch_size,
                torch_module=torch,
            )
            features[viewpoint_index] = encoded.astype(np.float16)
        _write_npz_atomic(
            path,
            {
                "viewpoint_ids": np.asarray(viewpoints),
                "features": features,
                **{
                    name: np.asarray(value)
                    for name, value in scan_provenance.items()
                },
            },
        )
        scan_records[scan] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "viewpoint_count": len(viewpoints),
        }
        print(
            f"[{scan_index}/{len(scans)}] {scan}: "
            f"{len(viewpoints)} viewpoints"
        )

    manifest = _base_manifest(
        args,
        "visual",
        feature_dim,
        model_provenance,
    )
    manifest.update(
        {
            "views_per_viewpoint": 36,
            "view_heading_degrees": [index * 30 for index in range(12)],
            "view_elevation_degrees": [-30, 0, 30],
            "camera": {
                "width": args.width,
                "height": args.height,
                "vfov_degrees": args.vfov,
            },
            "simulator_source_color_space": "BGR",
            "input_color_space": "RGB",
            "annotation_sha256": sha256_file(Path(args.annotation)),
            "scans": scan_records,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    print(f"Wrote visual CLIP cache for {len(scans)} scans to {output_dir}")


def verify_caches(args: argparse.Namespace) -> None:
    records = load_annotation_instructions(args.annotation)
    _check_expected_count(records, args.expected_count)
    instruction_store = InstructionCLIPFeatureStore(
        args.instruction_cache,
        expected_model_id=args.model_id,
        expected_model_revision=args.model_revision,
    )
    if len(instruction_store) != len(records):
        raise ValueError(
            f"Instruction cache has {len(instruction_store)} records, "
            f"expected {len(records)}"
        )
    for record in records:
        instruction_store(str(record["instr_id"]), str(record["instruction"]))

    visual_store = VisualCLIPFeatureStore(
        args.visual_cache_dir,
        expected_model_id=args.model_id,
        expected_model_revision=args.model_revision,
        max_cached_scans=1,
    )
    scans = sorted({str(record["scan"]) for record in records})
    if set(visual_store.scan_ids) != set(scans):
        raise ValueError(
            "Visual manifest scan set differs from the annotation scan set"
        )
    for scan in scans:
        viewpoints = _included_viewpoints(
            Path(args.connectivity_dir) / f"{scan}_connectivity.json"
        )
        for viewpoint in viewpoints:
            visual_store(
                {
                    "scan": scan,
                    "viewpoint": viewpoint,
                    "heading": 0.0,
                    "elevation": 0.0,
                }
            )
    if instruction_store.model_id != visual_store.model_id:
        raise ValueError("Instruction and visual caches use different CLIP models")
    if instruction_store.model_revision != visual_store.model_revision:
        raise ValueError("Instruction and visual caches use different revisions")
    if (
        instruction_store.model_weights_sha256
        != visual_store.model_weights_sha256
    ):
        raise ValueError(
            "Instruction and visual caches use different model weights"
        )
    if instruction_store.feature_dim != visual_store.feature_dim:
        raise ValueError("Instruction and visual feature dimensions differ")
    annotation_sha256 = sha256_file(Path(args.annotation))
    if (
        instruction_store.manifest.get("annotation_sha256")
        != annotation_sha256
        or visual_store.manifest.get("annotation_sha256")
        != annotation_sha256
    ):
        raise ValueError("CLIP cache annotation fingerprint mismatch")
    print(
        f"PASS {len(records)} instructions, {len(scans)} scans, "
        f"feature_dim={instruction_store.feature_dim}, model={args.model_id}"
    )


def _build_rendering_simulator(args: argparse.Namespace) -> Any:
    if not Path(args.scan_data_dir).exists():
        raise FileNotFoundError(
            f"Matterport3D scan data not found: {args.scan_data_dir}"
        )
    try:
        import MatterSim
    except ImportError as exc:
        raise ImportError(
            "MatterSim is required to render raw RGB views for CLIP"
        ) from exc
    simulator = MatterSim.Simulator()
    simulator.setDatasetPath(args.scan_data_dir)
    simulator.setNavGraphPath(args.connectivity_dir)
    simulator.setRenderingEnabled(True)
    simulator.setCameraResolution(args.width, args.height)
    simulator.setCameraVFOV(math.radians(args.vfov))
    simulator.setDiscretizedViewingAngles(True)
    simulator.setBatchSize(1)
    simulator.initialize()
    return simulator


def _render_36_views(
    simulator: Any,
    scan: str,
    viewpoint: str,
    image_module: Any,
) -> Sequence[Any]:
    images = []
    for view_index in range(36):
        if view_index == 0:
            simulator.newEpisode(
                [scan],
                [viewpoint],
                [0],
                [math.radians(-30)],
            )
        elif view_index % 12 == 0:
            simulator.makeAction([0], [1.0], [1.0])
        else:
            simulator.makeAction([0], [1.0], [0])
        state = simulator.getState()[0]
        if int(state.viewIndex) != view_index:
            raise RuntimeError(
                f"MatterSim view index mismatch: expected {view_index}, "
                f"got {state.viewIndex}"
            )
        bgr = np.asarray(state.rgb, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[-1] != 3:
            raise RuntimeError(
                f"MatterSim returned invalid BGR shape {bgr.shape} for "
                f"{scan}/{viewpoint}/view-{view_index}"
            )
        images.append(image_module.fromarray(_matterport_bgr_to_rgb(bgr)))
    return images


def _matterport_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert MatterSim's documented BGR buffer to contiguous RGB."""

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 BGR image, got {array.shape}")
    return array[..., ::-1].copy()


def _encode_images(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    *,
    device: str,
    dtype: Any,
    batch_size: int,
    torch_module: Any,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    features = []
    for offset in range(0, len(images), batch_size):
        batch = images[offset:offset + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {
            name: value.to(device=device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device)
            for name, value in inputs.items()
        }
        with torch_module.inference_mode():
            embeddings = model(**inputs).image_embeds
            embeddings = torch_module.nn.functional.normalize(embeddings, dim=-1)
        features.append(embeddings.float().cpu().numpy())
    return np.concatenate(features, axis=0)


def _included_viewpoints(path: Path) -> Sequence[str]:
    with path.open(encoding="utf-8") as file_obj:
        connectivity = json.load(file_obj)
    viewpoints = sorted(
        str(item["image_id"])
        for item in connectivity
        if item.get("included")
    )
    if not viewpoints:
        raise ValueError(f"No included viewpoints in {path}")
    return viewpoints


def _resume_scan_record(
    path: Path,
    expected_viewpoints: Sequence[str],
    feature_dim: int,
    expected_provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and reuse one completed scan after an interrupted build."""

    try:
        with np.load(path, allow_pickle=False) as cache:
            viewpoint_ids = np.asarray(cache["viewpoint_ids"]).astype(str)
            features = np.asarray(cache["features"], dtype=np.float32)
            actual_provenance = {
                name: _npz_scalar(cache, name)
                for name in expected_provenance
            }
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot resume invalid scan cache {path}; rerun with --overwrite"
        ) from exc
    if viewpoint_ids.tolist() != list(expected_viewpoints):
        raise ValueError(
            f"Viewpoint IDs changed for {path}; rerun with --overwrite"
        )
    if not _provenance_equal(actual_provenance, expected_provenance):
        raise ValueError(
            f"Cache provenance changed for {path}; rerun with --overwrite"
        )
    expected_shape = (len(expected_viewpoints), 36, feature_dim)
    if features.shape != expected_shape or not np.all(np.isfinite(features)):
        raise ValueError(
            f"Invalid feature matrix in {path}; rerun with --overwrite"
        )
    norms = np.linalg.norm(features, axis=-1)
    if np.any(norms <= 1e-6) or np.max(np.abs(norms - 1.0)) > 0.02:
        raise ValueError(
            f"Unnormalized feature matrix in {path}; rerun with --overwrite"
        )
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "viewpoint_count": len(expected_viewpoints),
    }


def _base_manifest(
    args: argparse.Namespace,
    cache_type: str,
    feature_dim: int,
    model_provenance: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": CLIP_CACHE_SCHEMA_VERSION,
        "cache_type": cache_type,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        **model_provenance,
        "feature_dim": int(feature_dim),
        "normalized": True,
        "storage_dtype": "float16",
    }


def _scan_provenance(
    args: argparse.Namespace,
    model_provenance: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": CLIP_CACHE_SCHEMA_VERSION,
        "model_id": str(args.model_id),
        "model_revision": str(args.model_revision),
        "model_weights_sha256": str(
            model_provenance["model_weights_sha256"]
        ),
        "input_color_space": "RGB",
        "camera_width": int(args.width),
        "camera_height": int(args.height),
        "camera_vfov_degrees": float(args.vfov),
    }


def _model_weight_provenance(model_path: Path) -> Dict[str, Any]:
    """Fingerprint the exact local CLIP weights, including sharded models."""

    if not model_path.is_dir():
        raise FileNotFoundError(f"CLIP model directory not found: {model_path}")
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        files = sorted(model_path.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(
            "No model*.safetensors or pytorch_model*.bin weights found in "
            f"{model_path}"
        )

    combined_digest = hashlib.sha256()
    file_records = []
    for path in files:
        file_digest = hashlib.sha256()
        relative_name = path.relative_to(model_path).as_posix()
        combined_digest.update(relative_name.encode("utf-8"))
        combined_digest.update(b"\0")
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                file_digest.update(chunk)
                combined_digest.update(chunk)
        combined_digest.update(b"\0")
        file_records.append(
            {"file": relative_name, "sha256": file_digest.hexdigest()}
        )
    return {
        "model_weights_sha256": combined_digest.hexdigest(),
        "model_weight_files": file_records,
    }


def _npz_scalar(cache: Any, name: str) -> Any:
    value = np.asarray(cache[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar array")
    return value.item()


def _provenance_equal(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
) -> bool:
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if isinstance(expected_value, float):
            try:
                if not math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _write_npz_atomic(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file_obj:
        np.savez(file_obj, **arrays)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)


def _check_expected_count(records: Sequence[Any], expected_count: int) -> None:
    if expected_count < 0:
        raise ValueError("expected_count must be nonnegative")
    if expected_count and len(records) != expected_count:
        raise ValueError(
            f"Annotation contains {len(records)} instructions, "
            f"expected {expected_count}"
        )


def _resolve_device(device: str, torch_module: Any) -> str:
    if device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "text":
        build_text_cache(args)
    elif args.command == "visual":
        build_visual_cache(args)
    elif args.command == "verify":
        verify_caches(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
