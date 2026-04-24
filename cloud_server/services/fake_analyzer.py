from __future__ import annotations

from typing import Optional


def _image_metadata(image_bytes: bytes, content_type: Optional[str], filename: Optional[str]) -> dict:
    metadata = {
        "bytes": len(image_bytes),
        "content_type": content_type or "",
        "filename": filename or "",
        "width": None,
        "height": None,
        "decoded": False,
    }

    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            metadata["height"], metadata["width"] = int(img.shape[0]), int(img.shape[1])
            metadata["decoded"] = True
    except Exception as exc:  # Stage A should keep running even if OpenCV is missing.
        metadata["decode_error"] = str(exc)

    return metadata


def analyze_fake(
    *,
    image_bytes: bytes,
    content_type: Optional[str],
    filename: Optional[str],
    track_id: int,
    frame_id: int,
    ts_ms: int,
    source: str,
    local_name: Optional[str],
    local_conf: Optional[float],
) -> dict:
    metadata = _image_metadata(image_bytes, content_type, filename)

    normalized_local_name = (local_name or "").strip()
    known = bool(normalized_local_name and normalized_local_name.lower() != "unknown")
    identity_name = normalized_local_name if known else "Unknown"
    identity_conf = float(local_conf) if local_conf is not None else (80.0 if known else 0.0)

    return {
        "ok": True,
        "track_id": track_id,
        "frame_id": frame_id,
        "ts_ms": ts_ms,
        "source": source,
        "identity": {
            "name": identity_name,
            "known": known,
            "confidence": max(0.0, min(99.0, identity_conf)),
            "distance": None,
        },
        "emotion": {
            "label": "Calm",
            "confidence": 55.0,
            "probs": {
                "Calm": 0.55,
                "Happy": 0.25,
                "Sad": 0.10,
                "Angry": 0.10,
            },
        },
        "image": metadata,
        "debug": {
            "mode": "stage_a_fake_analyzer",
            "note": "HTTP and GPU-provider plumbing only; real models are not connected yet.",
        },
    }
