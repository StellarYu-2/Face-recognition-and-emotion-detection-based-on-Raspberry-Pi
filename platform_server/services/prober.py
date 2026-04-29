from __future__ import annotations

import asyncio
import os
import socket
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .store import SQLiteStore


BroadcastCallback = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class DeviceProbe:
    device_id: str
    role: str
    display_name: str
    host: str
    port: int
    interval_ms: int = 5000
    timeout_ms: int = 1200


class PlatformProber:
    def __init__(self, store: SQLiteStore, probes: list[DeviceProbe], broadcast: BroadcastCallback) -> None:
        self._store = store
        self._probes = probes
        self._broadcast = broadcast
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def probes(self) -> list[DeviceProbe]:
        return list(self._probes)

    async def start(self) -> None:
        if not self._probes or self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="platform-prober")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        # One shared interval is enough for Stage B.
        interval_s = max(1.0, min(probe.interval_ms for probe in self._probes) / 1000.0)
        while not self._stop.is_set():
            await asyncio.gather(*(self._probe_once(probe) for probe in self._probes))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                continue

    async def _probe_once(self, probe: DeviceProbe) -> None:
        reachable = await asyncio.to_thread(_tcp_reachable, probe.host, probe.port, probe.timeout_ms / 1000.0)
        payload = {
            "device_id": probe.device_id,
            "role": probe.role,
            "display_name": probe.display_name,
            "online": reachable,
            "merge_status": True,
            "status": {
                "connectivity": "reachable" if reachable else "unreachable",
                "probe_host": probe.host,
                "probe_port": probe.port,
                "probe": "tcp",
            },
            "metadata": {
                "source": "platform_probe",
            },
            "ts_ms": int(time.time() * 1000),
        }
        device = self._store.upsert_status(payload)
        await self._broadcast({"type": "device_status", "device": device})


def default_probes() -> list[DeviceProbe]:
    enabled = os.getenv("ASDUN_PROBE_PI_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return []
    return [
        DeviceProbe(
            device_id=os.getenv("ASDUN_PI_DEVICE_ID", "pi-01"),
            role=os.getenv("ASDUN_PI_ROLE", "raspberry_pi"),
            display_name=os.getenv("ASDUN_PI_DISPLAY_NAME", "asdun@asdun"),
            host=os.getenv("ASDUN_PI_HOST", "asdun"),
            port=int(os.getenv("ASDUN_PI_PROBE_PORT", "22")),
            interval_ms=int(os.getenv("ASDUN_PI_PROBE_INTERVAL_MS", "5000")),
            timeout_ms=int(os.getenv("ASDUN_PI_PROBE_TIMEOUT_MS", "1200")),
        )
    ]


def _tcp_reachable(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
