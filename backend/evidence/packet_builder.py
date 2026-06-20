from __future__ import annotations

from uuid import uuid4

from ..models.schemas import EvidenceAsset, ReviewStatus, ViolationPacket, ViolationType


def build_candidate_packet(
    violation_type: ViolationType,
    confidence: float,
    timestamp_seconds: float,
    zone_name: str,
    evidence_paths: list[str],
    plate_text: str | None = None,
) -> ViolationPacket:
    evidence = [
        EvidenceAsset(kind="annotated_frame" if path.lower().endswith((".jpg", ".png")) else "clip", path=path)
        for path in evidence_paths
    ]
    return ViolationPacket(
        packet_id=uuid4().hex,
        violation_type=violation_type,
        confidence=confidence,
        timestamp_seconds=timestamp_seconds,
        zone_name=zone_name,
        plate_text=plate_text,
        review_status=ReviewStatus.pending,
        evidence=evidence,
    )
