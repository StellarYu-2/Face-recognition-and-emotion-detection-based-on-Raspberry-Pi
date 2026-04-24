from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "cloud_server" / "config.yaml"


@dataclass(frozen=True)
class EmotionConfig:
    enabled: bool
    model_path: Path
    input_size: int
    input_layout: str
    non_calm_floor: float
    handoff_margin: float
    sad_floor: float
    sad_handoff_margin: float
    sad_vs_other_margin: float
    preferred_provider: str
    fallback_provider: str


@dataclass(frozen=True)
class IdentityConfig:
    enabled: bool
    backend: str
    model_path: Path
    insightface_root: Path
    insightface_model_name: str
    insightface_det_width: int
    insightface_det_height: int
    insightface_det_score: float
    input_size: int
    input_layout: str
    mean: float
    norm: float
    match_threshold: float
    margin_threshold: float
    score_top_k: int
    template_count: int
    score_topk_weight: float
    score_centroid_weight: float
    score_template_weight: float
    enroll_min_samples: int
    enroll_max_intra_distance: float
    enroll_outlier_mad_scale: float
    enroll_cross_person_margin: float
    enroll_cross_person_min_distance: float
    preferred_provider: str
    fallback_provider: str


@dataclass(frozen=True)
class GalleryConfig:
    store_path: Path
    images_root: Path


@dataclass(frozen=True)
class CloudConfig:
    emotion: EmotionConfig
    identity: IdentityConfig
    gallery: GalleryConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


@lru_cache(maxsize=1)
def load_config() -> CloudConfig:
    config_path = Path(os.environ.get("ASDUN_CLOUD_CONFIG", DEFAULT_CONFIG_PATH))
    raw = _load_yaml(config_path)
    runtime = raw.get("runtime", {}) if isinstance(raw.get("runtime", {}), dict) else {}
    emotion = raw.get("emotion", {}) if isinstance(raw.get("emotion", {}), dict) else {}
    identity = raw.get("identity", {}) if isinstance(raw.get("identity", {}), dict) else {}
    gallery = raw.get("gallery", {}) if isinstance(raw.get("gallery", {}), dict) else {}

    return CloudConfig(
        emotion=EmotionConfig(
            enabled=bool(emotion.get("enabled", True)),
            model_path=_resolve_path(str(emotion.get("model_path", "./models/emotion-ferplus-8.onnx"))),
            input_size=int(emotion.get("input_size", 64)),
            input_layout=str(emotion.get("input_layout", "nchw_gray")),
            non_calm_floor=float(emotion.get("non_calm_floor", 0.22)),
            handoff_margin=float(emotion.get("handoff_margin", 0.08)),
            sad_floor=float(emotion.get("sad_floor", 0.16)),
            sad_handoff_margin=float(emotion.get("sad_handoff_margin", 0.20)),
            sad_vs_other_margin=float(emotion.get("sad_vs_other_margin", 0.03)),
            preferred_provider=str(runtime.get("preferred_provider", "CUDAExecutionProvider")),
            fallback_provider=str(runtime.get("fallback_provider", "CPUExecutionProvider")),
        ),
        identity=IdentityConfig(
            enabled=bool(identity.get("enabled", True)),
            backend=str(identity.get("backend", "arcface_crop")).strip().lower(),
            model_path=_resolve_path(str(identity.get("model_path", "./models/arcfaceresnet100-8.onnx"))),
            insightface_root=_resolve_path(str(identity.get("insightface_root", "./models/insightface"))),
            insightface_model_name=str(identity.get("insightface_model_name", "buffalo_l")),
            insightface_det_width=int(identity.get("insightface_det_width", 640)),
            insightface_det_height=int(identity.get("insightface_det_height", 640)),
            insightface_det_score=float(identity.get("insightface_det_score", 0.45)),
            input_size=int(identity.get("input_size", 112)),
            input_layout=str(identity.get("input_layout", "nchw_rgb")),
            mean=float(identity.get("mean", 127.5)),
            norm=float(identity.get("norm", 0.0078125)),
            match_threshold=float(identity.get("match_threshold", 0.72)),
            margin_threshold=float(identity.get("margin_threshold", 0.08)),
            score_top_k=max(1, int(identity.get("score_top_k", 3))),
            template_count=max(1, int(identity.get("template_count", 4))),
            score_topk_weight=float(identity.get("score_topk_weight", 0.55)),
            score_centroid_weight=float(identity.get("score_centroid_weight", 0.30)),
            score_template_weight=float(identity.get("score_template_weight", 0.15)),
            enroll_min_samples=max(1, int(identity.get("enroll_min_samples", 3))),
            enroll_max_intra_distance=float(identity.get("enroll_max_intra_distance", 0.42)),
            enroll_outlier_mad_scale=float(identity.get("enroll_outlier_mad_scale", 3.5)),
            enroll_cross_person_margin=float(identity.get("enroll_cross_person_margin", 0.03)),
            enroll_cross_person_min_distance=float(identity.get("enroll_cross_person_min_distance", 0.16)),
            preferred_provider=str(runtime.get("preferred_provider", "CUDAExecutionProvider")),
            fallback_provider=str(runtime.get("fallback_provider", "CPUExecutionProvider")),
        ),
        gallery=GalleryConfig(
            store_path=_resolve_path(str(gallery.get("store_path", "./data/cloud_gallery.npz"))),
            images_root=_resolve_path(str(gallery.get("images_root", "./data/gallery"))),
        ),
    )
