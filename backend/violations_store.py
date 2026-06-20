"""
In-memory violation store — thread-safe for demo purposes.
Stores all detected violation packets and supports CRUD + analytics queries.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

from .models.schemas import ReviewAction, ReviewStatus, ViolationPacket, ViolationType


class ViolationsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packets: dict[str, ViolationPacket] = {}
        self._timeline: list[dict[str, Any]] = []  # ordered log of events

    # ------------------------------------------------------------------ write
    def add(self, packet: ViolationPacket) -> None:
        with self._lock:
            self._packets[packet.packet_id] = packet
            self._timeline.append(
                {
                    "event": "detected",
                    "packet_id": packet.packet_id,
                    "violation_type": packet.violation_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": packet.confidence,
                    "zone": packet.zone_name,
                }
            )

    def apply_review(self, action: ReviewAction) -> ViolationPacket | None:
        with self._lock:
            packet = self._packets.get(action.packet_id)
            if packet is None:
                return None
            status_map = {
                "approve": ReviewStatus.approved,
                "reject": ReviewStatus.rejected,
                "flag": ReviewStatus.flagged,
            }
            updated = packet.model_copy(
                update={"review_status": status_map[action.action]}
            )
            self._packets[action.packet_id] = updated
            self._timeline.append(
                {
                    "event": action.action,
                    "packet_id": action.packet_id,
                    "officer_id": action.officer_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return updated

    # ------------------------------------------------------------------ read
    def get(self, packet_id: str) -> ViolationPacket | None:
        with self._lock:
            return self._packets.get(packet_id)

    def list_all(
        self,
        status: ReviewStatus | None = None,
        violation_type: ViolationType | None = None,
        limit: int = 100,
    ) -> list[ViolationPacket]:
        with self._lock:
            packets = list(self._packets.values())
        if status is not None:
            packets = [p for p in packets if p.review_status == status]
        if violation_type is not None:
            packets = [p for p in packets if p.violation_type == violation_type]
        # newest first
        return packets[-limit:][::-1]

    # ------------------------------------------------------------------ analytics
    def get_analytics(self) -> dict[str, Any]:
        with self._lock:
            packets = list(self._packets.values())

        total = len(packets)
        by_type: dict[str, int] = defaultdict(int)
        by_status: dict[str, int] = defaultdict(int)
        by_zone: dict[str, int] = defaultdict(int)
        by_hour: dict[str, int] = defaultdict(int)
        confidence_buckets = [0, 0, 0, 0, 0]  # 0-20,20-40,40-60,60-80,80-100

        for p in packets:
            by_type[p.violation_type.value] += 1
            by_status[p.review_status.value] += 1
            by_zone[p.zone_name] += 1
            hour_key = str(int(p.timestamp_seconds // 3600) % 24).zfill(2) + ":00"
            by_hour[hour_key] += 1
            bucket = min(int(p.confidence * 5), 4)
            confidence_buckets[bucket] += 1

        pending = sum(1 for p in packets if p.review_status == ReviewStatus.pending)
        approved = sum(1 for p in packets if p.review_status == ReviewStatus.approved)

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
            "recent_events": self._timeline[-20:][::-1],
        }

    def clear(self) -> None:
        with self._lock:
            self._packets.clear()
            self._timeline.clear()


# Module-level singleton — shared across all API requests
store = ViolationsStore()
