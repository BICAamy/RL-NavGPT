"""Validated CLIP feature stores used by navigation rewards."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


CLIP_CACHE_SCHEMA_VERSION = 2
VISUAL_MANIFEST_NAME = "manifest.json"

VIEW_HEADINGS = np.asarray(
    [math.radians(index * 30) for _ in range(3) for index in range(12)],
    dtype=np.float32,
)
VIEW_ELEVATIONS = np.asarray(
    [math.radians(elevation) for elevation in (-30, 0, 30) for _ in range(12)],
    dtype=np.float32,
)


class CLIPFeatureCacheError(ValueError):
    """Raised when a CLIP cache is missing, corrupt, or incompatible."""


class InstructionCLIPFeatureStore:
    """Instruction feature cache keyed by exact ``instr_id`` and text hash."""

    def __init__(
        self,
        path: str,
        *,
        expected_model_id: Optional[str] = None,
        expected_model_revision: Optional[str] = None,
    ):
        self.path = Path(path)
        self.manifest_path = Path(f"{self.path}.manifest.json")
        self.manifest = _load_manifest(self.manifest_path, "instruction")
        _validate_expected_model(
            self.manifest,
            expected_model_id,
            expected_model_revision,
        )
        if not self.path.is_file():
            raise FileNotFoundError(f"Instruction CLIP cache not found: {self.path}")
        expected_cache_sha256 = str(self.manifest.get("cache_sha256", ""))
        if not _is_sha256(expected_cache_sha256):
            raise CLIPFeatureCacheError(
                f"Invalid instruction cache SHA256 in {self.manifest_path}"
            )
        if sha256_file(self.path) != expected_cache_sha256:
            raise CLIPFeatureCacheError(f"SHA256 mismatch for {self.path}")
        with np.load(self.path, allow_pickle=False) as cache:
            required = {"instr_ids", "instruction_sha256", "features"}
            missing = required.difference(cache.files)
            if missing:
                raise CLIPFeatureCacheError(
                    f"{self.path} is missing arrays: {sorted(missing)}"
                )
            instr_ids = np.asarray(cache["instr_ids"]).astype(str)
            instruction_hashes = np.asarray(
                cache["instruction_sha256"]
            ).astype(str)
            features = np.asarray(cache["features"], dtype=np.float32)

        if instr_ids.ndim != 1 or instruction_hashes.shape != instr_ids.shape:
            raise CLIPFeatureCacheError("Instruction cache ID/hash arrays are invalid")
        if features.ndim != 2 or features.shape[0] != instr_ids.size:
            raise CLIPFeatureCacheError("Instruction cache feature shape is invalid")
        if len(set(instr_ids.tolist())) != instr_ids.size:
            raise CLIPFeatureCacheError(
                "Instruction cache contains duplicate instr_id values"
            )
        _validate_feature_matrix(features, "instruction features")
        _validate_manifest_counts(
            self.manifest,
            record_count=instr_ids.size,
            feature_dim=features.shape[1],
        )

        self.model_id = str(self.manifest["model_id"])
        self.model_revision = str(self.manifest["model_revision"])
        self.model_weights_sha256 = str(
            self.manifest["model_weights_sha256"]
        )
        self.feature_dim = int(features.shape[1])
        self._features = features
        self._indices = {
            instr_id: index for index, instr_id in enumerate(instr_ids.tolist())
        }
        self._instruction_hashes = {
            instr_id: instruction_hashes[index]
            for index, instr_id in enumerate(instr_ids.tolist())
        }

    def __len__(self) -> int:
        return len(self._indices)

    def __call__(self, instr_id: str, instruction: str) -> np.ndarray:
        instr_id = str(instr_id)
        if instr_id not in self._indices:
            raise KeyError(f"Instruction CLIP cache is missing instr_id={instr_id}")
        actual_hash = sha256_text(str(instruction))
        expected_hash = self._instruction_hashes[instr_id]
        if actual_hash != expected_hash:
            raise CLIPFeatureCacheError(
                f"Instruction text mismatch for instr_id={instr_id}"
            )
        return self._features[self._indices[instr_id]].copy()


class VisualCLIPFeatureStore:
    """Lazy per-scan cache selecting the nearest of 36 raw RGB view features."""

    def __init__(
        self,
        directory: str,
        *,
        expected_model_id: Optional[str] = None,
        expected_model_revision: Optional[str] = None,
        max_cached_scans: int = 2,
    ):
        if max_cached_scans <= 0:
            raise ValueError("max_cached_scans must be positive")
        self.directory = Path(directory)
        self.manifest = _load_manifest(
            self.directory / VISUAL_MANIFEST_NAME,
            "visual",
        )
        _validate_expected_model(
            self.manifest,
            expected_model_id,
            expected_model_revision,
        )
        if int(self.manifest.get("views_per_viewpoint", -1)) != 36:
            raise CLIPFeatureCacheError(
                "Visual CLIP cache must contain 36 views per viewpoint"
            )
        scans = self.manifest.get("scans")
        if not isinstance(scans, Mapping) or not scans:
            raise CLIPFeatureCacheError("Visual manifest has no scan records")
        self.model_id = str(self.manifest["model_id"])
        self.model_revision = str(self.manifest["model_revision"])
        self.model_weights_sha256 = str(
            self.manifest["model_weights_sha256"]
        )
        self.feature_dim = int(self.manifest["feature_dim"])
        if self.manifest.get("input_color_space") != "RGB":
            raise CLIPFeatureCacheError(
                "Visual CLIP cache input_color_space must be RGB"
            )
        camera = self.manifest.get("camera")
        if not isinstance(camera, Mapping):
            raise CLIPFeatureCacheError("Visual manifest has no camera metadata")
        try:
            width = int(camera["width"])
            height = int(camera["height"])
            vfov_degrees = float(camera["vfov_degrees"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CLIPFeatureCacheError(
                "Visual manifest camera metadata is invalid"
            ) from exc
        if width <= 0 or height <= 0 or not 0.0 < vfov_degrees < 180.0:
            raise CLIPFeatureCacheError(
                "Visual manifest camera metadata is out of range"
            )
        self.camera = {
            "width": width,
            "height": height,
            "vfov_degrees": vfov_degrees,
        }
        self.max_cached_scans = max_cached_scans
        invalid_records = [
            str(key)
            for key, value in scans.items()
            if not isinstance(value, Mapping)
        ]
        if invalid_records:
            raise CLIPFeatureCacheError(
                "Visual manifest contains invalid scan records: "
                f"{invalid_records[:5]}"
            )
        self._scan_records = {
            str(key): dict(value) for key, value in scans.items()
        }
        self._scan_cache: OrderedDict[str, tuple[Dict[str, int], np.ndarray]] = (
            OrderedDict()
        )

    @property
    def scan_ids(self) -> Sequence[str]:
        return tuple(sorted(self._scan_records))

    def __call__(self, observation: Mapping[str, Any]) -> np.ndarray:
        scan = str(observation["scan"])
        viewpoint = str(observation["viewpoint"])
        indices, features = self._load_scan(scan)
        if viewpoint not in indices:
            raise KeyError(
                f"Visual CLIP cache is missing {scan}/{viewpoint}"
            )
        view_index = nearest_view_index(
            float(observation["heading"]),
            float(observation["elevation"]),
        )
        return features[indices[viewpoint], view_index].copy()

    def _load_scan(self, scan: str) -> tuple[Dict[str, int], np.ndarray]:
        if scan in self._scan_cache:
            value = self._scan_cache.pop(scan)
            self._scan_cache[scan] = value
            return value
        if scan not in self._scan_records:
            raise KeyError(f"Visual CLIP manifest is missing scan={scan}")
        record = self._scan_records[scan]
        filename = record.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CLIPFeatureCacheError(f"Invalid visual cache filename for {scan}")
        path = self.directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Visual CLIP scan cache not found: {path}")
        expected_sha256 = str(record.get("sha256", ""))
        if not _is_sha256(expected_sha256):
            raise CLIPFeatureCacheError(
                f"Invalid visual cache SHA256 for scan={scan}"
            )
        if sha256_file(path) != expected_sha256:
            raise CLIPFeatureCacheError(f"SHA256 mismatch for {path}")
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "viewpoint_ids",
                "features",
                "schema_version",
                "model_id",
                "model_revision",
                "model_weights_sha256",
                "input_color_space",
                "camera_width",
                "camera_height",
                "camera_vfov_degrees",
            }
            missing = required.difference(cache.files)
            if missing:
                raise CLIPFeatureCacheError(
                    f"{path} is missing arrays: {sorted(missing)}"
                )
            viewpoint_ids = np.asarray(cache["viewpoint_ids"]).astype(str)
            features = np.asarray(cache["features"], dtype=np.float32)
            provenance = {
                "schema_version": _npz_scalar(cache, "schema_version"),
                "model_id": _npz_scalar(cache, "model_id"),
                "model_revision": _npz_scalar(cache, "model_revision"),
                "model_weights_sha256": _npz_scalar(
                    cache,
                    "model_weights_sha256",
                ),
                "input_color_space": _npz_scalar(cache, "input_color_space"),
                "camera_width": _npz_scalar(cache, "camera_width"),
                "camera_height": _npz_scalar(cache, "camera_height"),
                "camera_vfov_degrees": _npz_scalar(
                    cache,
                    "camera_vfov_degrees",
                ),
            }
        expected_provenance = {
            "schema_version": CLIP_CACHE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_weights_sha256": self.model_weights_sha256,
            "input_color_space": "RGB",
            "camera_width": self.camera["width"],
            "camera_height": self.camera["height"],
            "camera_vfov_degrees": self.camera["vfov_degrees"],
        }
        if not _provenance_equal(provenance, expected_provenance):
            raise CLIPFeatureCacheError(
                f"Per-scan provenance does not match manifest for {path}"
            )
        if viewpoint_ids.ndim != 1:
            raise CLIPFeatureCacheError(f"Invalid viewpoint IDs in {path}")
        expected_shape = (viewpoint_ids.size, 36, self.feature_dim)
        if features.shape != expected_shape:
            raise CLIPFeatureCacheError(
                f"Expected visual features {expected_shape}, got {features.shape}"
            )
        if len(set(viewpoint_ids.tolist())) != viewpoint_ids.size:
            raise CLIPFeatureCacheError(f"Duplicate viewpoint IDs in {path}")
        _validate_feature_matrix(
            features.reshape(-1, self.feature_dim),
            f"visual features in {path}",
        )
        expected_count = int(record.get("viewpoint_count", -1))
        if expected_count != viewpoint_ids.size:
            raise CLIPFeatureCacheError(
                f"Viewpoint count mismatch for scan={scan}"
            )
        value = (
            {
                viewpoint: index
                for index, viewpoint in enumerate(viewpoint_ids.tolist())
            },
            features,
        )
        self._scan_cache[scan] = value
        while len(self._scan_cache) > self.max_cached_scans:
            self._scan_cache.popitem(last=False)
        return value


class CLIPTextFeatureEncoder:
    """Shared, cached CLIP text tower for online thought-quality scoring."""

    def __init__(
        self,
        model_path: str,
        *,
        model_id: str = "openai/clip-vit-large-patch14",
        model_revision: str = "main",
        device: str = "auto",
        dtype: str = "fp32",
        cache_size: int = 20_000,
        local_files_only: bool = True,
    ):
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("dtype must be fp32, fp16, or bf16")
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.device = _resolve_device(device, torch)
        if self.device == "cpu" and dtype != "fp32":
            raise ValueError("CPU CLIP text encoding requires dtype=fp32")
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            dtype=torch_dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.model_revision = model_revision
        self.feature_dim = int(self.model.config.projection_dim)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __call__(self, text: str) -> np.ndarray:
        return self.encode_many([text], batch_size=1)[0]

    def encode_many(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cleaned = [str(text).strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("CLIP text inputs must be non-empty")
        missing = list(
            dict.fromkeys(text for text in cleaned if text not in self._cache)
        )
        # Keep this call's results independently of the bounded cross-call LRU.
        # Otherwise a batch larger than cache_size evicts an item before the
        # output array is assembled.
        resolved: Dict[str, np.ndarray] = {
            text: self._cache[text]
            for text in cleaned
            if text in self._cache
        }
        if missing:
            import torch

            for offset in range(0, len(missing), batch_size):
                batch = missing[offset:offset + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {
                    name: value.to(self.device)
                    for name, value in inputs.items()
                }
                with torch.inference_mode():
                    embeddings = self.model(**inputs).text_embeds
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                arrays = embeddings.float().cpu().numpy()
                for text, feature in zip(batch, arrays):
                    array = feature.astype(np.float32, copy=True)
                    resolved[text] = array
                    self._cache[text] = array
                    self._cache.move_to_end(text)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        output = []
        for text in cleaned:
            feature = resolved[text]
            if text in self._cache:
                self._cache.move_to_end(text)
            output.append(feature.copy())
        return np.stack(output)


def nearest_view_index(heading: float, elevation: float) -> int:
    """Return the closest discretized MatterSim view index in angular space."""

    heading_deltas = np.arctan2(
        np.sin(VIEW_HEADINGS - heading),
        np.cos(VIEW_HEADINGS - heading),
    )
    elevation_deltas = VIEW_ELEVATIONS - elevation
    return int(np.argmin(heading_deltas ** 2 + elevation_deltas ** 2))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path, cache_type: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CLIP manifest not found: {path}")
    with path.open(encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if not isinstance(manifest, dict):
        raise CLIPFeatureCacheError(f"{path} must contain a JSON object")
    if manifest.get("schema_version") != CLIP_CACHE_SCHEMA_VERSION:
        raise CLIPFeatureCacheError(f"Unsupported CLIP cache schema in {path}")
    if manifest.get("cache_type") != cache_type:
        raise CLIPFeatureCacheError(
            f"Expected cache_type={cache_type} in {path}"
        )
    if not str(manifest.get("model_id", "")).strip():
        raise CLIPFeatureCacheError(f"Missing model_id in {path}")
    if not str(manifest.get("model_revision", "")).strip():
        raise CLIPFeatureCacheError(f"Missing model_revision in {path}")
    model_weights_sha256 = str(
        manifest.get("model_weights_sha256", "")
    )
    if not _is_sha256(model_weights_sha256):
        raise CLIPFeatureCacheError(
            f"Invalid model_weights_sha256 in {path}"
        )
    if manifest.get("normalized") is not True:
        raise CLIPFeatureCacheError(f"Cache is not marked normalized in {path}")
    if not _is_sha256(str(manifest.get("annotation_sha256", ""))):
        raise CLIPFeatureCacheError(f"Invalid annotation_sha256 in {path}")
    feature_dim = manifest.get("feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise CLIPFeatureCacheError(f"Invalid feature_dim in {path}")
    return manifest


def _validate_expected_model(
    manifest: Mapping[str, Any],
    expected_model_id: Optional[str],
    expected_model_revision: Optional[str],
) -> None:
    if expected_model_id and manifest["model_id"] != expected_model_id:
        raise CLIPFeatureCacheError(
            "CLIP model mismatch: expected "
            f"{expected_model_id}, got {manifest['model_id']}"
        )
    if (
        expected_model_revision
        and manifest["model_revision"] != expected_model_revision
    ):
        raise CLIPFeatureCacheError(
            "CLIP revision mismatch: expected "
            f"{expected_model_revision}, got {manifest['model_revision']}"
        )


def _npz_scalar(cache: Any, name: str) -> Any:
    value = np.asarray(cache[name])
    if value.ndim != 0:
        raise CLIPFeatureCacheError(f"{name} must be a scalar array")
    return value.item()


def _provenance_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if isinstance(expected_value, float):
            try:
                equal = math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            except (TypeError, ValueError):
                return False
            if not equal:
                return False
        elif actual_value != expected_value:
            return False
    return True


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _validate_manifest_counts(
    manifest: Mapping[str, Any],
    *,
    record_count: int,
    feature_dim: int,
) -> None:
    if int(manifest.get("record_count", -1)) != record_count:
        raise CLIPFeatureCacheError("CLIP cache record_count mismatch")
    if int(manifest["feature_dim"]) != feature_dim:
        raise CLIPFeatureCacheError("CLIP cache feature_dim mismatch")


def _validate_feature_matrix(features: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(features)):
        raise CLIPFeatureCacheError(f"{name} contain non-finite values")
    norms = np.linalg.norm(features, axis=-1)
    if np.any(norms <= 1e-6):
        raise CLIPFeatureCacheError(f"{name} contain zero-norm rows")
    if np.max(np.abs(norms - 1.0)) > 0.02:
        raise CLIPFeatureCacheError(f"{name} are not L2 normalized")


def _resolve_device(device: str, torch_module: Any) -> str:
    if device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
