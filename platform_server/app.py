from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    from .services.store import SQLiteStore
    from .services.prober import PlatformProber, default_probes
except ImportError:  # Allows `uvicorn app:app` from inside platform_server/.
    from services.store import SQLiteStore
    from services.prober import PlatformProber, default_probes


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ASDUN_PLATFORM_DB", ROOT / "data" / "asdun_platform.sqlite"))
ONLINE_TTL_MS = int(os.getenv("ASDUN_PLATFORM_ONLINE_TTL_MS", "30000"))

app = FastAPI(
    title="ASDUN Platform Server",
    version="0.1.0",
    description="Device status, recognition events, and WebSocket dashboard for ASDUN.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SQLiteStore(DB_PATH, online_ttl_ms=ONLINE_TTL_MS)


class WebSocketHub:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections)
        if not connections:
            return

        stale: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        if stale:
            async with self._lock:
                for websocket in stale:
                    self._connections.discard(websocket)


hub = WebSocketHub()
prober = PlatformProber(store, default_probes(), hub.broadcast)


@app.on_event("startup")
async def startup() -> None:
    await prober.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await prober.stop()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = ROOT / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "asdun-platform-server",
        "db_path": str(DB_PATH),
        "device_count": store.device_count(),
        "online_ttl_ms": ONLINE_TTL_MS,
        "probes": [
            {
                "device_id": probe.device_id,
                "display_name": probe.display_name,
                "host": probe.host,
                "port": probe.port,
            }
            for probe in prober.probes
        ],
    }


@app.get("/api/snapshot")
def snapshot() -> dict[str, Any]:
    return {"ok": True, **store.snapshot()}


@app.get("/api/devices")
def devices() -> dict[str, Any]:
    return {"ok": True, "devices": store.list_devices()}


@app.get("/api/status/latest")
def latest_status() -> dict[str, Any]:
    return {"ok": True, "devices": store.list_devices()}


@app.get("/api/people")
def people() -> dict[str, Any]:
    return {"ok": True, "people": store.list_people_profiles()}


@app.post("/api/status")
async def post_status(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        device = store.upsert_status(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"ok": False, "error": str(exc)}) from exc

    message = {"type": "device_status", "device": device}
    await hub.broadcast(message)
    return {"ok": True, "device": device}


@app.get("/api/events/recognition")
def recognition_events(
    limit: int = Query(100, ge=1, le=500),
    person: str | None = Query(None),
) -> dict[str, Any]:
    return {"ok": True, "events": store.list_recognition_events(limit=limit, person=person)}


@app.post("/api/events/recognition")
async def post_recognition_event(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    event = store.insert_recognition_event(payload)
    await hub.broadcast({"type": "recognition_event", "event": event})
    await hub.broadcast({"type": "people", "people": store.list_people_profiles()})
    return {"ok": True, "event": event}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    try:
        await websocket.send_json({"type": "snapshot", **store.snapshot()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(websocket)
    except Exception:
        await hub.disconnect(websocket)
        raise
