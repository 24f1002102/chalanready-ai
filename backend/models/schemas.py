from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ViolationType(str, Enum):
    wrong_side       = "wrong_side_driving"
    illegal_parking  = "illegal_or_footpath_parking"
    footpath_riding  = "footpath_riding"
    helmet           = "helmet_non_compliance"
    stopline         = "stop_line_violation"
    red_light        = "red_light_violation"
    triple_riding    = "triple_riding"
    seatbelt         = "seatbelt_non_compliance"


class ReviewStatus(str, Enum):
    pending  = "pending"
    approved = "approved"
    rejected = "rejected"
    flagged  = "flagged_for_re_review"


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionObservation(BaseModel):
    track_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    frame_index: int


class EvidenceAsset(BaseModel):
    kind: Literal["annotated_frame", "clip"]
    path: str


class ViolationPacket(BaseModel):
    packet_id: str
    violation_type: ViolationType
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp_seconds: float
    zone_name: str
    gps_lat: float = 12.9716   # Bengaluru default — will be per-camera in production
    gps_lng: float = 77.5946
    plate_text: str | None = None
    review_status: ReviewStatus = ReviewStatus.pending
    evidence: list[EvidenceAsset] = Field(default_factory=list)
    officer_notes: str | None = None


class ReviewAction(BaseModel):
    packet_id: str
    action: Literal["approve", "reject", "flag"]
    officer_id: str = "demo-officer"
    notes: str | None = None
