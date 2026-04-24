from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class GalleryPerson:
    name: str
    samples: int


class GalleryStore:
    def __init__(self, store_path: Path):
        self._store_path = store_path
        self._names: List[str] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self.reload()

    @property
    def count(self) -> int:
        return len(self._names)

    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def people(self) -> List[GalleryPerson]:
        counts: Dict[str, int] = {}
        for name in self._names:
            counts[name] = counts.get(name, 0) + 1
        return [GalleryPerson(name=name, samples=samples) for name, samples in sorted(counts.items())]

    def diagnostics(
        self,
        max_intra_distance: float = 0.42,
        cross_warning_distance: float = 0.16,
    ) -> Dict:
        people = self.people()
        if self._embeddings.size == 0 or not people:
            return {
                "embedding_dim": 0,
                "people": [],
                "pairs": [],
                "warning": "empty_gallery",
            }

        by_name: Dict[str, np.ndarray] = {}
        for name in sorted({person.name for person in people}):
            indices = [i for i, existing_name in enumerate(self._names) if existing_name == name]
            by_name[name] = self._embeddings[indices].astype(np.float32)

        person_rows = []
        for name, embeddings in by_name.items():
            centroid = self._normalize_embedding(np.mean(embeddings, axis=0))
            intra: Optional[float] = None
            max_intra: Optional[float] = None
            outliers = []
            if embeddings.shape[0] > 1:
                distances = 1.0 - np.clip(embeddings @ centroid, -1.0, 1.0)
                intra = float(np.mean(distances))
                max_intra = float(np.max(distances))
                outliers = [
                    {
                        "sample": int(idx),
                        "distance_to_centroid": round(float(distance), 6),
                    }
                    for idx, distance in enumerate(distances)
                    if float(distance) > max_intra_distance
                ]
            person_rows.append(
                {
                    "name": name,
                    "samples": int(embeddings.shape[0]),
                    "mean_distance_to_centroid": round(intra, 6) if intra is not None else None,
                    "max_distance_to_centroid": round(max_intra, 6) if max_intra is not None else None,
                    "outliers": outliers,
                }
            )

        pairs = []
        names = sorted(by_name.keys())
        for i, left_name in enumerate(names):
            left = by_name[left_name]
            left_centroid = self._normalize_embedding(np.mean(left, axis=0))
            for right_name in names[i + 1 :]:
                right = by_name[right_name]
                right_centroid = self._normalize_embedding(np.mean(right, axis=0))
                centroid_distance = float(1.0 - np.clip(left_centroid @ right_centroid, -1.0, 1.0))
                cross = 1.0 - np.clip(left @ right.T, -1.0, 1.0)
                closest = np.unravel_index(int(np.argmin(cross)), cross.shape)
                pairs.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "centroid_distance": round(centroid_distance, 6),
                        "min_sample_distance": round(float(np.min(cross)), 6),
                        "mean_sample_distance": round(float(np.mean(cross)), 6),
                        "closest_left_sample": int(closest[0]),
                        "closest_right_sample": int(closest[1]),
                        "warning": (
                            "too_close"
                            if centroid_distance < cross_warning_distance or float(np.min(cross)) < cross_warning_distance
                            else None
                        ),
                    }
                )

        warning = None
        if any(pair["warning"] for pair in pairs) or any(person["outliers"] for person in person_rows):
            warning = "some_people_are_too_close_or_gallery_may_be_contaminated"

        return {
            "embedding_dim": int(self._embeddings.shape[1]) if self._embeddings.ndim == 2 else 0,
            "people": person_rows,
            "pairs": pairs,
            "warning": warning,
        }

    def reload(self) -> None:
        if not self._store_path.exists():
            self._names = []
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            return

        loaded = np.load(self._store_path, allow_pickle=False)
        self._names = [str(name) for name in loaded["names"].tolist()]
        self._embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
        if self._embeddings.ndim != 2 or len(self._names) != self._embeddings.shape[0]:
            self._names = []
            self._embeddings = np.empty((0, 0), dtype=np.float32)

    def add_person(self, name: str, embeddings: Iterable[np.ndarray], replace: bool = True) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("empty_person_name")

        new_embeddings = [self._normalize_embedding(e) for e in embeddings]
        if not new_embeddings:
            raise ValueError("no_valid_embeddings")

        if replace:
            keep_indices = [i for i, existing_name in enumerate(self._names) if existing_name != clean_name]
            kept_names = [self._names[i] for i in keep_indices]
            kept_embeddings = self._embeddings[keep_indices] if keep_indices and self._embeddings.size else np.empty(
                (0, new_embeddings[0].shape[0]),
                dtype=np.float32,
            )
        else:
            kept_names = list(self._names)
            kept_embeddings = self._embeddings

        appended = np.vstack(new_embeddings).astype(np.float32)
        if kept_embeddings.size == 0:
            merged_embeddings = appended
        else:
            merged_embeddings = np.vstack([kept_embeddings, appended]).astype(np.float32)

        self._names = kept_names + [clean_name] * appended.shape[0]
        self._embeddings = merged_embeddings
        self._save()
        return appended.shape[0]

    def remove_person(self, name: str) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("empty_person_name")

        if not self._names:
            return 0

        keep_indices = [i for i, existing_name in enumerate(self._names) if existing_name != clean_name]
        removed = len(self._names) - len(keep_indices)
        if removed <= 0:
            return 0

        embedding_dim = self._embeddings.shape[1] if self._embeddings.ndim == 2 else 0
        self._names = [self._names[i] for i in keep_indices]
        self._embeddings = (
            self._embeddings[keep_indices].astype(np.float32)
            if keep_indices and self._embeddings.size
            else np.empty((0, embedding_dim), dtype=np.float32)
        )
        self._save()
        return removed

    def _save(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._store_path,
            names=np.asarray(self._names, dtype="<U128"),
            embeddings=self._embeddings.astype(np.float32),
        )

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-9:
            raise ValueError("zero_embedding")
        return arr / norm
