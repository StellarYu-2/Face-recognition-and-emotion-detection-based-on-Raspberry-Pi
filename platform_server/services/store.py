from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


class SQLiteStore:
    def __init__(self, db_path: Path, online_ttl_ms: int = 30_000) -> None:
        self.db_path = db_path
        self.online_ttl_ms = online_ttl_ms
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS devices (
                  device_id TEXT PRIMARY KEY,
                  role TEXT NOT NULL DEFAULT 'unknown',
                  display_name TEXT NOT NULL DEFAULT '',
                  online INTEGER NOT NULL DEFAULT 1,
                  last_seen_ms INTEGER NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_ms INTEGER NOT NULL,
                  updated_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS device_status (
                  device_id TEXT PRIMARY KEY,
                  status_json TEXT NOT NULL,
                  ts_ms INTEGER NOT NULL,
                  updated_ms INTEGER NOT NULL,
                  FOREIGN KEY(device_id) REFERENCES devices(device_id)
                );

                CREATE TABLE IF NOT EXISTS status_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  device_id TEXT NOT NULL,
                  status_json TEXT NOT NULL,
                  ts_ms INTEGER NOT NULL,
                  received_ms INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_status_events_device_ts
                  ON status_events(device_id, ts_ms DESC);

                CREATE TABLE IF NOT EXISTS recognition_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source_device TEXT NOT NULL,
                  track_id INTEGER,
                  frame_id INTEGER,
                  name TEXT,
                  known INTEGER,
                  identity_confidence REAL,
                  emotion TEXT,
                  emotion_confidence REAL,
                  latency_ms REAL,
                  raw_json TEXT NOT NULL,
                  ts_ms INTEGER NOT NULL,
                  received_ms INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_recognition_events_ts
                  ON recognition_events(ts_ms DESC);
                """
            )
            self._conn.commit()

    def upsert_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        device_id = str(payload.get("device_id", "")).strip()
        if not device_id:
            raise ValueError("device_id is required")

        received_ms = now_ms()
        ts_ms = int(payload.get("ts_ms") or received_ms)
        role = str(payload.get("role") or payload.get("device_role") or "unknown").strip() or "unknown"
        display_name = str(payload.get("display_name") or device_id).strip()
        online = bool(payload.get("online", True))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        status = payload.get("status")
        if not isinstance(status, dict):
            ignored = {"device_id", "role", "device_role", "display_name", "online", "ts_ms", "metadata", "merge_status"}
            status = {key: value for key, value in payload.items() if key not in ignored}
        status = dict(status)
        if status.get("probe") == "tcp":
            status.setdefault("network_online", online)
            status["network_seen_ms"] = received_ms
        if status.get("app") == "asdun_access":
            status.setdefault("app_online", online)
            status["app_seen_ms"] = received_ms
        if status.get("service") == "inference":
            status.setdefault("service_online", online)
            status["service_seen_ms"] = received_ms
        status.setdefault("online", online)
        metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            existing = self._conn.execute(
                "SELECT d.last_seen_ms, s.status_json FROM devices d LEFT JOIN device_status s ON s.device_id = d.device_id WHERE d.device_id = ?",
                (device_id,),
            ).fetchone()
            last_seen_ms = received_ms if online or existing is None else int(existing["last_seen_ms"])
            if bool(payload.get("merge_status", False)) and existing is not None:
                merged_status = _loads_json(existing["status_json"] or "{}")
                merged_status.update(status)
                status = merged_status
            status_json = json.dumps(status, ensure_ascii=False, separators=(",", ":"))

            self._conn.execute(
                """
                INSERT INTO devices(device_id, role, display_name, online, last_seen_ms, metadata_json, created_ms, updated_ms)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                  role=excluded.role,
                  display_name=excluded.display_name,
                  online=excluded.online,
                  last_seen_ms=excluded.last_seen_ms,
                  metadata_json=excluded.metadata_json,
                  updated_ms=excluded.updated_ms
                """,
                (device_id, role, display_name, int(online), last_seen_ms, metadata_json, received_ms, received_ms),
            )
            self._conn.execute(
                """
                INSERT INTO device_status(device_id, status_json, ts_ms, updated_ms)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                  status_json=excluded.status_json,
                  ts_ms=excluded.ts_ms,
                  updated_ms=excluded.updated_ms
                """,
                (device_id, status_json, ts_ms, received_ms),
            )
            self._conn.execute(
                "INSERT INTO status_events(device_id, status_json, ts_ms, received_ms) VALUES(?, ?, ?, ?)",
                (device_id, status_json, ts_ms, received_ms),
            )
            self._conn.commit()

        return self.latest_status_for(device_id)

    def latest_status_for(self, device_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT d.device_id, d.role, d.display_name, d.online, d.last_seen_ms,
                       d.metadata_json, d.created_ms, d.updated_ms,
                       s.status_json, s.ts_ms AS status_ts_ms
                FROM devices d
                LEFT JOIN device_status s ON s.device_id = d.device_id
                WHERE d.device_id = ?
                """,
                (device_id,),
            ).fetchone()
        if row is None:
            raise KeyError(device_id)
        return self._device_row_to_dict(row)

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT d.device_id, d.role, d.display_name, d.online, d.last_seen_ms,
                       d.metadata_json, d.created_ms, d.updated_ms,
                       s.status_json, s.ts_ms AS status_ts_ms
                FROM devices d
                LEFT JOIN device_status s ON s.device_id = d.device_id
                ORDER BY d.role, d.device_id
                """
            ).fetchall()
        return [self._device_row_to_dict(row) for row in rows]

    def insert_recognition_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_device = str(payload.get("source_device") or payload.get("device_id") or "unknown").strip()
        received_ms = now_ms()
        ts_ms = int(payload.get("ts_ms") or received_ms)
        identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
        emotion = payload.get("emotion") if isinstance(payload.get("emotion"), dict) else {}

        name = payload.get("name") or identity.get("name")
        known = payload.get("known")
        if known is None:
            known = identity.get("known")
        identity_confidence = payload.get("identity_confidence")
        if identity_confidence is None:
            identity_confidence = identity.get("confidence")
        emotion_label = payload.get("emotion") if isinstance(payload.get("emotion"), str) else emotion.get("label")
        emotion_confidence = payload.get("emotion_confidence")
        if emotion_confidence is None:
            emotion_confidence = emotion.get("confidence")

        raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO recognition_events(
                  source_device, track_id, frame_id, name, known, identity_confidence,
                  emotion, emotion_confidence, latency_ms, raw_json, ts_ms, received_ms
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_device,
                    payload.get("track_id"),
                    payload.get("frame_id"),
                    name,
                    None if known is None else int(bool(known)),
                    identity_confidence,
                    emotion_label,
                    emotion_confidence,
                    payload.get("latency_ms"),
                    raw_json,
                    ts_ms,
                    received_ms,
                ),
            )
            self._conn.commit()
            event_id = int(cur.lastrowid)
        return self.get_recognition_event(event_id)

    def get_recognition_event(self, event_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM recognition_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._recognition_row_to_dict(row)

    def list_recognition_events(self, limit: int = 100, person: str | None = None) -> list[dict[str, Any]]:
        safe_limit = max(1, min(500, int(limit)))
        person_name = str(person or "").strip()
        with self._lock:
            if person_name:
                rows = self._conn.execute(
                    """
                    SELECT * FROM recognition_events
                    WHERE name IS NOT NULL
                      AND LOWER(TRIM(name)) = LOWER(?)
                    ORDER BY ts_ms DESC, id DESC
                    LIMIT ?
                    """,
                    (person_name, safe_limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM recognition_events ORDER BY ts_ms DESC, id DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
        return [self._recognition_row_to_dict(row) for row in rows]

    def list_people_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM recognition_events
                WHERE known = 1
                  AND name IS NOT NULL
                  AND TRIM(name) != ''
                  AND LOWER(TRIM(name)) != 'unknown'
                ORDER BY ts_ms ASC, id ASC
                """
            ).fetchall()

        profiles: dict[str, dict[str, Any]] = {}
        for row in rows:
            event = self._recognition_row_to_dict(row)
            name = str(event.get("name") or "").strip()
            if not name:
                continue

            profile = profiles.setdefault(
                name,
                {
                    "name": name,
                    "event_count": 0,
                    "identity_confidence_sum": 0.0,
                    "first_seen_ms": None,
                    "last_seen_ms": None,
                    "sources": set(),
                    "emotions": {},
                },
            )
            profile["event_count"] += 1
            if event.get("identity_confidence") is not None:
                profile["identity_confidence_sum"] += _clamp_pct(event["identity_confidence"])
            if event.get("source_device"):
                profile["sources"].add(str(event["source_device"]))
            ts_ms = int(event.get("ts_ms") or event.get("received_ms") or 0)
            if ts_ms:
                profile["first_seen_ms"] = ts_ms if profile["first_seen_ms"] is None else min(profile["first_seen_ms"], ts_ms)
                profile["last_seen_ms"] = ts_ms if profile["last_seen_ms"] is None else max(profile["last_seen_ms"], ts_ms)

            emotion = str(event.get("emotion") or "").strip()
            if emotion:
                bucket = profile["emotions"].setdefault(emotion, {"count": 0, "confidence_sum": 0.0})
                bucket["count"] += 1
                bucket["confidence_sum"] += _clamp_pct(event.get("emotion_confidence"))

        people: list[dict[str, Any]] = []
        for profile in profiles.values():
            total = max(1, int(profile["event_count"]))
            emotions = []
            for label, values in profile["emotions"].items():
                count = int(values["count"])
                confidence_sum = float(values["confidence_sum"])
                emotions.append(
                    {
                        "label": label,
                        "count": count,
                        "weighted_pct": count * 100.0 / total,
                        "avg_confidence": confidence_sum / max(1, count),
                    }
                )
            emotions.sort(key=lambda item: (-item["weighted_pct"], item["label"]))
            people.append(
                {
                    "name": profile["name"],
                    "event_count": total,
                    "avg_identity_confidence": profile["identity_confidence_sum"] / total,
                    "first_seen_ms": profile["first_seen_ms"],
                    "last_seen_ms": profile["last_seen_ms"],
                    "sources": sorted(profile["sources"]),
                    "dominant_emotion": emotions[0] if emotions else None,
                    "emotions": emotions,
                }
            )
        people.sort(key=lambda item: (-(item["last_seen_ms"] or 0), item["name"]))
        return people

    def snapshot(self) -> dict[str, Any]:
        return {
            "devices": self.list_devices(),
            "people": self.list_people_profiles(),
            "recognition_events": self.list_recognition_events(limit=50),
            "server_time_ms": now_ms(),
        }

    def device_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS count FROM devices").fetchone()
        return int(row["count"])

    def _device_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        now = now_ms()
        last_seen_ms = int(row["last_seen_ms"])
        stale = now - last_seen_ms > self.online_ttl_ms
        stored_online = bool(row["online"])
        status = _loads_json(row["status_json"] or "{}")
        online = stored_online and not stale
        return {
            "device_id": row["device_id"],
            "role": row["role"],
            "display_name": row["display_name"] or row["device_id"],
            "online": online,
            "stale": stale,
            "last_seen_ms": last_seen_ms,
            "metadata": _loads_json(row["metadata_json"] or "{}"),
            "status": status,
            "signals": _derive_signals(
                role=str(row["role"] or ""),
                online=online,
                status=status,
                now=now,
                ttl_ms=self.online_ttl_ms,
            ),
            "status_ts_ms": row["status_ts_ms"],
            "created_ms": row["created_ms"],
            "updated_ms": row["updated_ms"],
        }

    @staticmethod
    def _recognition_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source_device": row["source_device"],
            "track_id": row["track_id"],
            "frame_id": row["frame_id"],
            "name": row["name"],
            "known": None if row["known"] is None else bool(row["known"]),
            "identity_confidence": row["identity_confidence"],
            "emotion": row["emotion"],
            "emotion_confidence": row["emotion_confidence"],
            "latency_ms": row["latency_ms"],
            "raw": _loads_json(row["raw_json"] or "{}"),
            "ts_ms": row["ts_ms"],
            "received_ms": row["received_ms"],
        }


def _loads_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp_pct(value: Any) -> float:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, pct))


def _fresh_bool(status: dict[str, Any], value_key: str, seen_key: str, now: int, ttl_ms: int) -> bool | None:
    if value_key not in status:
        return None
    seen_ms = status.get(seen_key)
    if isinstance(seen_ms, (int, float)) and now - int(seen_ms) > ttl_ms:
        return False
    return bool(status.get(value_key))


def _signal(state: str, label: str, detail: str = "") -> dict[str, str]:
    return {"state": state, "label": label, "detail": detail}


def _derive_signals(role: str, online: bool, status: dict[str, Any], now: int, ttl_ms: int) -> dict[str, dict[str, str]]:
    network_online = _fresh_bool(status, "network_online", "network_seen_ms", now, ttl_ms)
    app_online = _fresh_bool(status, "app_online", "app_seen_ms", now, ttl_ms)
    service_online = _fresh_bool(status, "service_online", "service_seen_ms", now, ttl_ms)

    if network_online is None:
        network_online = online
    if service_online is None and role == "inference_server":
        service_online = online

    cloud_connected = status.get("cloud_connected")
    inference_ready = bool(status.get("emotion_ready")) and bool(status.get("identity_ready"))
    provider = str(status.get("provider") or status.get("device") or "")

    network = _signal("ok", "Online", str(status.get("probe_host") or "")) if network_online else _signal("bad", "Offline")

    if role == "raspberry_pi":
        if app_online is True:
            mode = str(status.get("mode") or "running")
            app = _signal("ok", "Running", mode)
        elif app_online is False:
            app = _signal("bad", "Stopped")
        else:
            app = _signal("warn", "Unknown")

        if app_online is True:
            cloud = _signal("ok", "Connected") if bool(cloud_connected) else _signal("bad", "Disconnected")
        else:
            cloud = _signal("warn", "No app")
        inference = _signal("muted", "-", "")
    elif role == "inference_server":
        app = _signal("ok", "Running", str(status.get("service") or "inference")) if service_online else _signal("bad", "Stopped")
        cloud = _signal("muted", "-", "")
        if service_online and inference_ready:
            inference = _signal("ok", "Ready", provider)
        elif service_online:
            inference = _signal("warn", "Partial", provider)
        else:
            inference = _signal("bad", "Offline")
    else:
        app = _signal("warn", "Unknown")
        cloud = _signal("muted", "-", "")
        inference = _signal("muted", "-", "")

    return {
        "network": network,
        "app": app,
        "cloud_link": cloud,
        "inference": inference,
    }
