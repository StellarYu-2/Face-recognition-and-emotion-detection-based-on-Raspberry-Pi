from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import IdentityConfig
from .gallery_store import GalleryStore
from .ort_providers import build_provider_request, provider_names


@dataclass(frozen=True)
class IdentityStatus:
    enabled: bool
    ready: bool
    backend: str
    model_path: str
    provider: str
    available_providers: List[str]
    requested_providers: List[str]
    session_providers: List[str]
    input_name: str
    output_name: str
    error: Optional[str]
    warning: Optional[str]
    gallery_count: int


@dataclass(frozen=True)
class PersonProfile:
    name: str
    samples: np.ndarray
    centroid: np.ndarray
    templates: np.ndarray
    original_samples: int
    kept_samples: int


class IdentityService:
    def __init__(self, config: IdentityConfig, gallery: GalleryStore):
        self._config = config
        self._gallery = gallery
        self._session = None
        self._insightface_app = None
        self._input_name = ""
        self._output_name = ""
        self._provider = "none"
        self._available_providers: List[str] = []
        self._requested_providers: List[str] = []
        self._session_providers: List[str] = []
        self._error: Optional[str] = None
        self._warning: Optional[str] = None

        if not config.enabled:
            self._error = "disabled_by_config"
            return
        if config.backend == "insightface":
            self._init_insightface()
            return
        self._init_arcface_crop()

    def _init_arcface_crop(self) -> None:
        if not self._config.model_path.exists():
            self._error = f"model_not_found: {self._config.model_path}"
            return

        try:
            import onnxruntime as ort

            self._available_providers = list(ort.get_available_providers())
            providers = build_provider_request(
                self._config.preferred_provider,
                self._config.fallback_provider,
                self._available_providers,
            )
            self._requested_providers = provider_names(providers)
            if not providers:
                raise RuntimeError(f"no usable onnxruntime providers: {self._available_providers}")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self._config.model_path),
                sess_options=session_options,
                providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._session_providers = list(self._session.get_providers())
            self._provider = self._session_providers[0] if self._session_providers else "none"
            if self._config.preferred_provider in self._available_providers and self._config.preferred_provider not in self._session_providers:
                self._warning = (
                    f"preferred provider {self._config.preferred_provider} was available but session used "
                    f"{self._session_providers}"
                )
        except Exception as exc:
            self._session = None
            self._error = str(exc)

    def _init_insightface(self) -> None:
        try:
            model_dir = self._config.insightface_root / "models" / self._config.insightface_model_name
            if not model_dir.exists():
                raise RuntimeError(f"insightface_model_dir_not_found: {model_dir}")

            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            preload = getattr(ort, "preload_dlls", None)
            if preload is not None:
                try:
                    preload(directory="")
                except Exception:
                    pass

            self._available_providers = list(ort.get_available_providers())
            providers = build_provider_request(
                self._config.preferred_provider,
                self._config.fallback_provider,
                self._available_providers,
            )
            self._requested_providers = provider_names(providers)
            if not providers:
                raise RuntimeError(f"no usable onnxruntime providers: {self._available_providers}")

            self._insightface_app = FaceAnalysis(
                name=self._config.insightface_model_name,
                root=str(self._config.insightface_root),
                providers=self._requested_providers,
                allowed_modules=["detection", "recognition"],
            )
            ctx_id = 0 if "CUDAExecutionProvider" in self._requested_providers else -1
            self._insightface_app.prepare(
                ctx_id=ctx_id,
                det_size=(self._config.insightface_det_width, self._config.insightface_det_height),
                det_thresh=self._config.insightface_det_score,
            )
            self._session_providers = self._requested_providers
            self._provider = self._session_providers[0] if self._session_providers else "none"
            self._input_name = "insightface.FaceAnalysis"
            self._output_name = "face.embedding"
        except Exception as exc:
            self._insightface_app = None
            self._error = str(exc)

    @property
    def ready(self) -> bool:
        return self._session is not None or self._insightface_app is not None

    def status(self) -> IdentityStatus:
        return IdentityStatus(
            enabled=self._config.enabled,
            ready=self.ready,
            backend=self._config.backend,
            model_path=self._model_path_for_status(),
            provider=self._provider,
            available_providers=self._available_providers,
            requested_providers=self._requested_providers,
            session_providers=self._session_providers,
            input_name=self._input_name,
            output_name=self._output_name,
            error=self._error,
            warning=self._warning,
            gallery_count=self._gallery.count,
        )

    def _model_path_for_status(self) -> str:
        if self._config.backend == "insightface":
            return str(self._config.insightface_root / "models" / self._config.insightface_model_name)
        return str(self._config.model_path)

    def enroll(self, name: str, images: List[bytes], replace: bool = True) -> Dict:
        if not self.ready:
            raise RuntimeError(self._error or "identity_service_not_ready")

        embeddings = []
        errors = []
        for idx, image_bytes in enumerate(images):
            try:
                embeddings.append(self.extract_embedding(image_bytes))
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})

        if not embeddings:
            return {
                "ok": False,
                "name": name,
                "samples": 0,
                "skipped": errors,
                "error": "no_valid_embeddings",
                "gallery_count": self._gallery.count,
            }

        try:
            cleaned_embeddings, quality = self._clean_enrollment_embeddings(name, embeddings)
        except ValueError as exc:
            return {
                "ok": False,
                "name": name,
                "samples": 0,
                "skipped": errors,
                "error": str(exc),
                "gallery_count": self._gallery.count,
            }

        added = self._gallery.add_person(name, cleaned_embeddings, replace=replace)
        return {
            "ok": True,
            "name": name,
            "samples": added,
            "skipped": errors,
            "quality": quality,
            "gallery_count": self._gallery.count,
        }

    def match(self, image_bytes: bytes) -> Dict:
        if not self.ready:
            raise RuntimeError(self._error or "identity_service_not_ready")
        if self._gallery.count <= 0:
            return self._unknown("gallery_empty")

        embedding = self.extract_embedding(image_bytes)
        names = self._gallery.names
        gallery_embeddings = self._gallery.embeddings
        if gallery_embeddings.size == 0:
            return self._unknown("gallery_empty")

        profiles = self._build_profiles(names, gallery_embeddings)
        if not profiles:
            return self._unknown("empty_profiles")

        ranked = sorted(
            ((name, self._score_profile(embedding, profile)) for name, profile in profiles.items()),
            key=lambda item: item[1]["score"],
        )
        top_name, top_stats = ranked[0]
        top_score = top_stats["score"]
        top_nearest = top_stats["nearest"]
        second_name = ranked[1][0] if len(ranked) > 1 else None
        second_score = ranked[1][1]["score"] if len(ranked) > 1 else None
        gap = (second_score - top_score) if second_score is not None else 1.0
        distance_ok = top_score <= self._config.match_threshold
        gap_ok = gap >= self._config.margin_threshold
        known = distance_ok and gap_ok

        if not known:
            result = self._unknown("distance_or_gap_failed")
            result["top1"] = top_name
            result["top2"] = second_name
            result["distance"] = round(top_score, 6)
            result["nearest_distance"] = round(top_nearest, 6)
            result["second_distance"] = round(second_score, 6) if second_score is not None else None
            result["gap"] = round(gap, 6)
            result["samples"] = profiles[top_name].kept_samples
            result["debug"]["score_top_k"] = self._config.score_top_k
            result["debug"]["centroid_distance"] = round(top_stats["centroid"], 6)
            result["debug"]["template_distance"] = round(top_stats["template"], 6)
            return result

        confidence = self._confidence(top_score, gap)
        return {
            "name": top_name,
            "known": True,
            "confidence": round(confidence, 3),
            "distance": round(top_score, 6),
            "nearest_distance": round(top_nearest, 6),
            "top2": second_name,
            "second_distance": round(second_score, 6) if second_score is not None else None,
            "gap": round(gap, 6),
            "samples": profiles[top_name].kept_samples,
            "debug": {
                "provider": self._provider,
                "model": self._config.insightface_model_name if self._config.backend == "insightface" else self._config.model_path.name,
                "threshold": self._config.match_threshold,
                "margin_threshold": self._config.margin_threshold,
                "score_top_k": self._config.score_top_k,
                "centroid_distance": round(top_stats["centroid"], 6),
                "template_distance": round(top_stats["template"], 6),
                "original_samples": profiles[top_name].original_samples,
                "kept_samples": profiles[top_name].kept_samples,
            },
        }

    def _clean_enrollment_embeddings(self, name: str, embeddings: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        clean_name = name.strip()
        normalized = [self._normalize_embedding(embedding) for embedding in embeddings]
        if not normalized:
            raise ValueError("no_valid_embeddings")

        matrix = np.vstack(normalized).astype(np.float32)
        kept_indices, self_distances, limit = self._select_inliers(matrix)
        skipped = []
        for idx, distance in enumerate(self_distances):
            if idx not in kept_indices:
                skipped.append(
                    {
                        "index": idx,
                        "reason": "self_outlier",
                        "distance_to_self_centroid": round(float(distance), 6),
                        "limit": round(float(limit), 6),
                    }
                )

        kept = matrix[kept_indices] if kept_indices else np.empty((0, matrix.shape[1]), dtype=np.float32)
        if kept.shape[0] < min(self._config.enroll_min_samples, matrix.shape[0]):
            raise ValueError(
                f"too_few_clean_enrollment_samples: kept={kept.shape[0]} required={self._config.enroll_min_samples}"
            )

        other_profiles = self._build_profiles(self._gallery.names, self._gallery.embeddings, exclude_name=clean_name)
        if other_profiles:
            self_centroid = self._normalize_embedding(np.mean(kept, axis=0))
            final_indices = []
            for local_idx, embedding in enumerate(kept):
                self_distance = float(1.0 - np.clip(embedding @ self_centroid, -1.0, 1.0))
                best_other_name = None
                best_other_distance = 2.0
                for other_name, profile in other_profiles.items():
                    distance = float(1.0 - np.clip(embedding @ profile.centroid, -1.0, 1.0))
                    if distance < best_other_distance:
                        best_other_name = other_name
                        best_other_distance = distance

                conflicts_by_absolute_distance = best_other_distance < self._config.enroll_cross_person_min_distance
                conflicts_by_margin = best_other_distance + self._config.enroll_cross_person_margin < self_distance
                if conflicts_by_absolute_distance or conflicts_by_margin:
                    skipped.append(
                        {
                            "index": int(kept_indices[local_idx]),
                            "reason": "too_close_to_existing_person",
                            "other": best_other_name,
                            "distance_to_other_centroid": round(best_other_distance, 6),
                            "distance_to_self_centroid": round(self_distance, 6),
                        }
                    )
                else:
                    final_indices.append(local_idx)

            kept = kept[final_indices] if final_indices else np.empty((0, matrix.shape[1]), dtype=np.float32)
            if kept.shape[0] < min(self._config.enroll_min_samples, matrix.shape[0]):
                raise ValueError(
                    f"too_few_non_conflicting_samples: kept={kept.shape[0]} required={self._config.enroll_min_samples}"
                )

        centroid = self._normalize_embedding(np.mean(kept, axis=0))
        final_distances = 1.0 - np.clip(kept @ centroid, -1.0, 1.0)
        quality = {
            "accepted": int(kept.shape[0]),
            "received_embeddings": int(matrix.shape[0]),
            "skipped": skipped,
            "self_distance_limit": round(float(limit), 6),
            "mean_distance_to_centroid": round(float(np.mean(final_distances)), 6),
            "max_distance_to_centroid": round(float(np.max(final_distances)), 6),
        }
        return [embedding for embedding in kept], quality

    def _build_profiles(
        self,
        names: List[str],
        embeddings: np.ndarray,
        exclude_name: Optional[str] = None,
    ) -> Dict[str, PersonProfile]:
        if embeddings.size == 0 or not names:
            return {}

        grouped: Dict[str, List[np.ndarray]] = {}
        for name, embedding in zip(names, embeddings):
            clean_name = str(name)
            if exclude_name is not None and clean_name == exclude_name:
                continue
            grouped.setdefault(clean_name, []).append(self._normalize_embedding(embedding))

        profiles: Dict[str, PersonProfile] = {}
        for person_name, person_embeddings in grouped.items():
            matrix = np.vstack(person_embeddings).astype(np.float32)
            kept_indices, _, _ = self._select_inliers(matrix)
            clean = matrix[kept_indices] if kept_indices else matrix
            centroid = self._normalize_embedding(np.mean(clean, axis=0))
            templates = self._select_templates(clean, centroid)
            profiles[person_name] = PersonProfile(
                name=person_name,
                samples=clean,
                centroid=centroid,
                templates=templates,
                original_samples=int(matrix.shape[0]),
                kept_samples=int(clean.shape[0]),
            )
        return profiles

    def _score_profile(self, embedding: np.ndarray, profile: PersonProfile) -> Dict[str, float]:
        sample_distances = 1.0 - np.clip(profile.samples @ embedding, -1.0, 1.0)
        ordered = np.sort(sample_distances)
        k = min(self._config.score_top_k, ordered.shape[0])
        topk = float(np.mean(ordered[:k]))
        nearest = float(ordered[0])
        centroid = float(1.0 - np.clip(profile.centroid @ embedding, -1.0, 1.0))
        template = float(np.min(1.0 - np.clip(profile.templates @ embedding, -1.0, 1.0)))

        weight_total = (
            self._config.score_topk_weight
            + self._config.score_centroid_weight
            + self._config.score_template_weight
        )
        if weight_total <= 1e-6:
            score = topk
        else:
            score = (
                self._config.score_topk_weight * topk
                + self._config.score_centroid_weight * centroid
                + self._config.score_template_weight * template
            ) / weight_total

        return {
            "score": float(score),
            "nearest": nearest,
            "topk": topk,
            "centroid": centroid,
            "template": template,
            "k": float(k),
        }

    def _select_inliers(self, embeddings: np.ndarray) -> Tuple[List[int], np.ndarray, float]:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            return [], np.empty((0,), dtype=np.float32), 0.0

        centroid = self._normalize_embedding(np.mean(embeddings, axis=0))
        distances = 1.0 - np.clip(embeddings @ centroid, -1.0, 1.0)
        if embeddings.shape[0] < max(3, self._config.enroll_min_samples):
            return list(range(embeddings.shape[0])), distances.astype(np.float32), float(self._config.enroll_max_intra_distance)

        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        robust_sigma = 1.4826 * mad
        adaptive_limit = median + max(0.05, self._config.enroll_outlier_mad_scale * robust_sigma)
        limit = min(float(self._config.enroll_max_intra_distance), float(adaptive_limit))
        kept = [idx for idx, distance in enumerate(distances) if float(distance) <= limit]
        required = min(self._config.enroll_min_samples, embeddings.shape[0])
        if len(kept) < required:
            kept = [int(idx) for idx in np.argsort(distances)[:required]]
        return kept, distances.astype(np.float32), limit

    def _select_templates(self, embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        templates = [centroid]
        if embeddings.size == 0:
            return np.vstack(templates).astype(np.float32)

        distances_to_centroid = 1.0 - np.clip(embeddings @ centroid, -1.0, 1.0)
        selected = [int(np.argmin(distances_to_centroid))]
        while len(selected) < min(self._config.template_count, embeddings.shape[0]):
            selected_embeddings = embeddings[selected]
            distance_to_selected = 1.0 - np.clip(embeddings @ selected_embeddings.T, -1.0, 1.0)
            min_distance = np.min(distance_to_selected, axis=1)
            for idx in selected:
                min_distance[idx] = -1.0
            next_idx = int(np.argmax(min_distance))
            if min_distance[next_idx] <= 0:
                break
            selected.append(next_idx)

        templates.extend(self._normalize_embedding(embeddings[idx]) for idx in selected)
        return np.vstack(templates).astype(np.float32)

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        if self._config.backend == "insightface":
            return self._extract_embedding_insightface(image_bytes)
        if self._session is None:
            raise RuntimeError(self._error or "identity_service_not_ready")
        tensor = self._preprocess(image_bytes)
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-9:
            raise RuntimeError("zero_embedding")
        return embedding / norm

    def _extract_embedding_insightface(self, image_bytes: bytes) -> np.ndarray:
        if self._insightface_app is None:
            raise RuntimeError(self._error or "insightface_not_ready")
        bgr = self._decode_bgr(image_bytes)
        faces = self._insightface_app.get(bgr)
        if not faces:
            raise RuntimeError("no_face_detected_by_insightface")

        def area(face) -> float:
            bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
            if bbox.size < 4:
                return 0.0
            return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

        face = max(faces, key=area)
        embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-9:
            raise RuntimeError("zero_embedding")
        return embedding / norm

    @staticmethod
    def _decode_bgr(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            raise RuntimeError("decode_failed")
        return bgr

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        bgr = self._decode_bgr(image_bytes)

        resized = cv2.resize(
            bgr,
            (self._config.input_size, self._config.input_size),
            interpolation=cv2.INTER_AREA if max(bgr.shape[:2]) > self._config.input_size else cv2.INTER_LINEAR,
        )
        if self._config.input_layout == "nchw_rgb":
            img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) - self._config.mean) * self._config.norm
            return np.transpose(img, (2, 0, 1))[None, :, :, :]

        img = (resized.astype(np.float32) - self._config.mean) * self._config.norm
        return np.transpose(img, (2, 0, 1))[None, :, :, :]

    def _confidence(self, distance: float, gap: float) -> float:
        distance_score = max(0.0, min(1.0, (self._config.match_threshold - distance) / self._config.match_threshold))
        gap_score = max(0.0, min(1.0, gap / max(self._config.margin_threshold * 3.0, 1e-6)))
        return 100.0 * (0.72 * distance_score + 0.28 * gap_score)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-9:
            raise ValueError("zero_embedding")
        return arr / norm

    @staticmethod
    def _unknown(reason: str) -> Dict:
        return {
            "name": "Unknown",
            "known": False,
            "confidence": 0.0,
            "distance": None,
            "gap": None,
            "samples": 0,
            "debug": {"reason": reason},
        }
