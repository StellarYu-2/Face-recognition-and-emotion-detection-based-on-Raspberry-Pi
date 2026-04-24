from __future__ import annotations

import time
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .services.config import load_config
    from .services.emotion_service import EmotionService
    from .services.fake_analyzer import analyze_fake
    from .services.gallery_store import GalleryStore
    from .services.identity_service import IdentityService
    from .services.runtime import get_runtime_info, preload_onnxruntime_dlls
except ImportError:  # Allows `uvicorn app:app` from inside cloud_server/.
    from services.config import load_config
    from services.emotion_service import EmotionService
    from services.fake_analyzer import analyze_fake
    from services.gallery_store import GalleryStore
    from services.identity_service import IdentityService
    from services.runtime import get_runtime_info, preload_onnxruntime_dlls


app = FastAPI(
    title="ASDUN Cloud Inference Server",
    version="0.1.0",
    description="Hybrid cloud inference server for Raspberry Pi face recognition and emotion analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CONFIG = load_config()
_ORT_PRELOAD_ERROR = preload_onnxruntime_dlls()
_GALLERY = GalleryStore(_CONFIG.gallery.store_path)
_EMOTION_SERVICE = EmotionService(_CONFIG.emotion)
_IDENTITY_SERVICE = IdentityService(_CONFIG.identity, _GALLERY)


def _unknown_identity(reason: str) -> dict:
    return {
        "name": "Unknown",
        "known": False,
        "confidence": 0.0,
        "distance": None,
        "gap": None,
        "samples": 0,
        "debug": {"reason": reason},
    }


@app.get("/health")
def health() -> dict:
    runtime = get_runtime_info()
    emotion_status = _EMOTION_SERVICE.status()
    identity_status = _IDENTITY_SERVICE.status()
    return {
        "ok": True,
        "service": "asdun-cloud-server",
        "stage": "C-cloud-identity",
        "device": runtime.device,
        "providers": runtime.providers,
        "active_provider": runtime.active_provider,
        "provider_error": runtime.provider_error,
        "preload_error": _ORT_PRELOAD_ERROR or runtime.preload_error,
        "gpu_name": runtime.gpu_name,
        "emotion": {
            "enabled": emotion_status.enabled,
            "ready": emotion_status.ready,
            "provider": emotion_status.provider,
            "available_providers": emotion_status.available_providers,
            "requested_providers": emotion_status.requested_providers,
            "session_providers": emotion_status.session_providers,
            "model_path": emotion_status.model_path,
            "input_name": emotion_status.input_name,
            "output_name": emotion_status.output_name,
            "error": emotion_status.error,
            "warning": emotion_status.warning,
        },
        "identity": {
            "enabled": identity_status.enabled,
            "ready": identity_status.ready,
            "backend": identity_status.backend,
            "provider": identity_status.provider,
            "available_providers": identity_status.available_providers,
            "requested_providers": identity_status.requested_providers,
            "session_providers": identity_status.session_providers,
            "model_path": identity_status.model_path,
            "input_name": identity_status.input_name,
            "output_name": identity_status.output_name,
            "error": identity_status.error,
            "warning": identity_status.warning,
        },
        "gallery_count": _GALLERY.count,
    }


@app.get("/gallery")
def gallery() -> dict:
    return {
        "ok": True,
        "gallery_count": _GALLERY.count,
        "people": [
            {"name": person.name, "samples": person.samples}
            for person in _GALLERY.people()
        ],
    }


@app.get("/gallery/diagnostics")
def gallery_diagnostics() -> dict:
    return {
        "ok": True,
        **_GALLERY.diagnostics(
            max_intra_distance=_CONFIG.identity.enroll_max_intra_distance,
            cross_warning_distance=_CONFIG.identity.enroll_cross_person_min_distance,
        ),
    }


@app.post("/gallery/reload")
def reload_gallery() -> dict:
    _GALLERY.reload()
    return gallery()


@app.post("/gallery/enroll")
async def enroll_gallery(
    name: str = Form(...),
    replace: bool = Form(True),
    images: List[UploadFile] = File(...),
) -> dict:
    image_bytes = [await image.read() for image in images]
    try:
        result = _IDENTITY_SERVICE.enroll(name=name, images=image_bytes, replace=replace)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "ok": False,
                "name": name.strip(),
                "error": str(exc),
                "gallery_count": _GALLERY.count,
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "ok": False,
                "name": name.strip(),
                "error": str(exc),
                "gallery_count": _GALLERY.count,
            },
        ) from exc

    if not result.get("ok", False):
        raise HTTPException(status_code=422, detail=result)
    return result


@app.post("/gallery/delete")
def delete_gallery_person(name: str = Form(...)) -> dict:
    removed = _GALLERY.remove_person(name)
    return {
        "ok": True,
        "name": name.strip(),
        "removed": removed,
        "gallery_count": _GALLERY.count,
    }


@app.post("/analyze")
async def analyze(
    track_id: int = Form(...),
    frame_id: int = Form(...),
    ts_ms: int = Form(...),
    source: str = Form("raspberry-pi"),
    local_name: Optional[str] = Form(None),
    local_conf: Optional[float] = Form(None),
    image: UploadFile = File(...),
) -> dict:
    start = time.perf_counter()
    image_bytes = await image.read()
    if not image_bytes:
        return {
            "ok": False,
            "track_id": track_id,
            "frame_id": frame_id,
            "error": "empty_image",
        }

    result = analyze_fake(
        image_bytes=image_bytes,
        content_type=image.content_type,
        filename=image.filename,
        track_id=track_id,
        frame_id=frame_id,
        ts_ms=ts_ms,
        source=source,
        local_name=local_name,
        local_conf=local_conf,
    )

    if _IDENTITY_SERVICE.ready:
        try:
            result["identity"] = _IDENTITY_SERVICE.match(image_bytes)
            result["debug"]["identity_mode"] = "stage_c_cloud_identity"
        except Exception as exc:
            result["debug"]["identity_error"] = str(exc)
            result["identity"] = _unknown_identity("identity_exception")
    else:
        result["debug"]["identity_error"] = _IDENTITY_SERVICE.status().error
        result["identity"] = _unknown_identity("identity_service_not_ready")

    if _EMOTION_SERVICE.ready:
        try:
            result["emotion"] = _EMOTION_SERVICE.infer(image_bytes)
            result["debug"]["mode"] = "stage_b_cloud_emotion"
        except Exception as exc:
            result["debug"]["emotion_error"] = str(exc)
    else:
        result["debug"]["emotion_error"] = _EMOTION_SERVICE.status().error

    result["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    return result
