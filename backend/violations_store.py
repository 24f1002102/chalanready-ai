"""
Violation store — SQLite-backed persistence.
Survives server restarts, cloud deployments, Railway sleeps.

Keeps the same API surface as the original in-memory store so zero
changes needed in routes.py or anywhere else.

DB file: PROJECT_ROOT/sample_data/outputs/chalanready.db
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models.schemas import ReviewAction, ReviewStatus, ViolationPacket, ViolationType


# Store database in generated outputs so local demo data is not committed.
_DB_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "outputs" / "chalanready.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    packet_id       TEXT PRIMARY KEY,
    violation_type  TEXT NOT NULL,
    confidence      REAL NOT NULL,
    timestamp_secs  REAL NOT NULL,
    zone_name       TEXT NOT NULL,
    plate_text      TEXT,
    review_status   TEXT NOT NULL DEFAULT 'pending',
    gps_lat         REAL DEFAULT 12.9716,
    gps_lng         REAL DEFAULT 77.5946,
    evidence_json   TEXT DEFAULT '[]',
    officer_notes   TEXT
);

CREATE TABLE IF NOT EXISTS timeline (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event           TEXT NOT NULL,
    packet_id       TEXT,
    officer_id      TEXT,
    ts              TEXT NOT NULL,
    confidence      REAL,
    zone            TEXT,
    violation_type  TEXT
);
"""


class ViolationsStore:
    """
    Thread-safe, SQLite-backed violation store.
    Compatible drop-in for the previous in-memory implementation.
    """

    def __init__(self, db_path: Path | str = _DB_PATH) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ write

    def add(self, packet: ViolationPacket) -> None:
        evidence_json = json.dumps([e.model_dump() for e in packet.evidence])
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO violations
                   (packet_id, violation_type, confidence, timestamp_secs,
                    zone_name, plate_text, review_status,
                    gps_lat, gps_lng, evidence_json, officer_notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    packet.packet_id,
                    packet.violation_type.value,
                    packet.confidence,
                    packet.timestamp_seconds,
                    packet.zone_name,
                    packet.plate_text,
                    packet.review_status.value,
                    packet.gps_lat,
                    packet.gps_lng,
                    evidence_json,
                    packet.officer_notes,
                ),
            )
            conn.execute(
                """INSERT INTO timeline (event, packet_id, ts, confidence, zone, violation_type)
                   VALUES (?,?,?,?,?,?)""",
                (
                    "detected",
                    packet.packet_id,
                    datetime.utcnow().isoformat(),
                    packet.confidence,
                    packet.zone_name,
                    packet.violation_type.value,
                ),
            )

    def apply_review(self, action: ReviewAction) -> ViolationPacket | None:
        status_map = {
            "approve": ReviewStatus.approved.value,
            "reject":  ReviewStatus.rejected.value,
            "flag":    ReviewStatus.flagged.value,
        }
        new_status = status_map[action.action]
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE violations SET review_status=? WHERE packet_id=?",
                (new_status, action.packet_id),
            )
            conn.execute(
                """INSERT INTO timeline (event, packet_id, officer_id, ts)
                   VALUES (?,?,?,?)""",
                (action.action, action.packet_id, action.officer_id, datetime.utcnow().isoformat()),
            )
        return self.get(action.packet_id)

    # ------------------------------------------------------------------ read

    def get(self, packet_id: str) -> ViolationPacket | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM violations WHERE packet_id=?", (packet_id,)
            ).fetchone()
        return self._row_to_packet(row) if row else None

    def list_all(
        self,
        status: ReviewStatus | None = None,
        violation_type: ViolationType | None = None,
        limit: int = 100,
    ) -> list[ViolationPacket]:
        clauses, params = [], []
        if status is not None:
            clauses.append("review_status=?")
            params.append(status.value)
        if violation_type is not None:
            clauses.append("violation_type=?")
            params.append(violation_type.value)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM violations {where} ORDER BY timestamp_secs DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_packet(r) for r in rows]

    # ------------------------------------------------------------------ analytics

    def get_analytics(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM violations").fetchall()
            timeline_rows = conn.execute(
                "SELECT * FROM timeline ORDER BY id DESC LIMIT 20"
            ).fetchall()

        packets = [self._row_to_packet(r) for r in rows]
        total = len(packets)
        by_type: dict[str, int] = defaultdict(int)
        by_status: dict[str, int] = defaultdict(int)
        by_zone: dict[str, int] = defaultdict(int)
        by_hour: dict[str, int] = defaultdict(int)
        confidence_buckets = [0, 0, 0, 0, 0]

        for p in packets:
            by_type[p.violation_type.value] += 1
            by_status[p.review_status.value] += 1
            by_zone[p.zone_name] += 1
            # Convert Unix timestamp to IST (UTC+5:30) for correct hour bucketing
            from datetime import datetime, timezone, timedelta
            _IST = timezone(timedelta(hours=5, minutes=30))
            dt_ist = datetime.fromtimestamp(p.timestamp_seconds, tz=_IST)
            hour_key = dt_ist.strftime("%H:00")
            by_hour[hour_key] += 1
            bucket = min(int(p.confidence * 5), 4)
            confidence_buckets[bucket] += 1

        pending  = sum(1 for p in packets if p.review_status == ReviewStatus.pending)
        approved = sum(1 for p in packets if p.review_status == ReviewStatus.approved)

        recent_events = [
            {
                "event": r["event"],
                "packet_id": r["packet_id"],
                "officer_id": r["officer_id"],
                "timestamp": r["ts"],
                "confidence": r["confidence"],
                "zone": r["zone"],
                "violation_type": r["violation_type"],
            }
            for r in timeline_rows
        ]

        return {
            "total_violations": total,
            "pending_review": pending,
            "approved_challans": approved,
            "approval_rate": round(approved / total * 100, 1) if total else 0,
            "by_type": dict(by_type),
            "by_status": dict(by_status),
            "by_zone": dict(by_zone),
            "by_hour": dict(by_hour),
            "confidence_distribution": confidence_buckets,
            "recent_events": recent_events,
        }

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM violations")
            conn.execute("DELETE FROM timeline")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_packet(row: sqlite3.Row) -> ViolationPacket:
        evidence = json.loads(row["evidence_json"] or "[]")
        from .models.schemas import EvidenceAsset
        evidence_assets = [EvidenceAsset(**e) for e in evidence]
        return ViolationPacket(
            packet_id=row["packet_id"],
            violation_type=ViolationType(row["violation_type"]),
            confidence=row["confidence"],
            timestamp_seconds=row["timestamp_secs"],
            zone_name=row["zone_name"],
            plate_text=row["plate_text"],
            review_status=ReviewStatus(row["review_status"]),
            gps_lat=row["gps_lat"] or 12.9716,
            gps_lng=row["gps_lng"] or 77.5946,
            evidence=evidence_assets,
            officer_notes=row["officer_notes"],
        )


# Module-level singleton — shared across all API requests
store = ViolationsStore()
