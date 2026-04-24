from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional


@dataclass(frozen=True)
class RuntimeInfo:
    providers: List[str]
    active_provider: str
    device: str
    provider_error: Optional[str]
    gpu_name: Optional[str]
    preload_error: Optional[str]


@lru_cache(maxsize=1)
def preload_onnxruntime_dlls() -> Optional[str]:
    try:
        import onnxruntime as ort

        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            return "onnxruntime.preload_dlls is not available"

        errors: List[str] = []

        # Importing torch first lets ONNX Runtime reuse CUDA/cuDNN DLLs bundled
        # with PyTorch wheels when that install route is chosen.
        try:
            import torch  # noqa: F401
        except Exception as exc:
            errors.append(f"torch preload skipped: {exc}")

        for kwargs in ({}, {"directory": ""}):
            try:
                preload(**kwargs)
                return None
            except Exception as exc:
                label = "default" if not kwargs else "nvidia_site_packages"
                errors.append(f"{label}: {exc}")

        return "; ".join(errors)
    except Exception as exc:
        return str(exc)


def _detect_gpu_name() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else None


def get_runtime_info() -> RuntimeInfo:
    provider_error = None
    providers: List[str] = []
    preload_error = preload_onnxruntime_dlls()

    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception as exc:
        provider_error = str(exc)

    if "CUDAExecutionProvider" in providers:
        active_provider = "CUDAExecutionProvider"
        device = "cuda"
    elif "DmlExecutionProvider" in providers:
        active_provider = "DmlExecutionProvider"
        device = "directml"
    elif "CPUExecutionProvider" in providers:
        active_provider = "CPUExecutionProvider"
        device = "cpu"
    else:
        active_provider = "none"
        device = "unknown"

    return RuntimeInfo(
        providers=providers,
        active_provider=active_provider,
        device=device,
        provider_error=provider_error,
        gpu_name=_detect_gpu_name(),
        preload_error=preload_error,
    )
