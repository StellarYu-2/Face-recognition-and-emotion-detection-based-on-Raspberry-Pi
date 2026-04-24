from __future__ import annotations

from typing import List


def provider_name(provider) -> str:
    return provider[0] if isinstance(provider, tuple) else str(provider)


def provider_names(providers: List) -> List[str]:
    return [provider_name(provider) for provider in providers]


def build_provider_request(preferred_provider: str, fallback_provider: str, available: List[str]) -> List:
    providers: List = []
    if preferred_provider == "CUDAExecutionProvider" and "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": 1,
                },
            )
        )
    elif preferred_provider in available:
        providers.append(preferred_provider)

    requested = provider_names(providers)
    if fallback_provider in available and fallback_provider not in requested:
        providers.append(fallback_provider)
        requested.append(fallback_provider)
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in requested:
        providers.append("CPUExecutionProvider")
    return providers
