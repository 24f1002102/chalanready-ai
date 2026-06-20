from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ..models.schemas import ReviewAction, ReviewStatus, ViolationType
from ..pipeline import process_video
from ..violations_store import store


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "sample_data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "sample_data" / "outputs"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"


# ──────────────────────────────────────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "prototype": "ChalanReady AI",
        "version": "2.0.0",
        "stage": "full_violation_pipeline",
        "enforcement_mode": "officer_review_required",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Video processing
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/videos/process")
async def process_uploaded_video(
    file: UploadFile = File(...),
    detector: str = "auto",
    zone_name: str = "Zone-A / MG Road",
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    if suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Upload a video file (.mp4 / .avi / .mov / .mkv).")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    output_path = OUTPUT_DIR / f"{upload_path.stem}_annotated.mp4"

    contents = await file.read()
    upload_path.write_bytes(contents)

    try:
        result = process_video(
            input_path=upload_path,
            output_path=output_path,
            detector_backend=detector,
            zone_name=zone_name,
            store=store,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "Video processed. Violation candidates require officer review.",
        "officer_review_required": True,
        "result": result,
        "violations_in_store": store.get_analytics()["total_violations"],
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Violations — CRUD
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/violations")
def list_violations(
    status: str | None = Query(None, description="pending | approved | rejected | flagged_for_re_review"),
    violation_type: str | None = Query(None),
    limit: int = Query(50, le=200),
) -> dict[str, Any]:
    status_enum = ReviewStatus(status) if status else None
    type_enum = ViolationType(violation_type) if violation_type else None

    packets = store.list_all(status=status_enum, violation_type=type_enum, limit=limit)
    return {
        "count": len(packets),
        "violations": [_packet_to_dict(p) for p in packets],
    }


@router.get("/violations/{packet_id}")
def get_violation(packet_id: str) -> dict[str, Any]:
    packet = store.get(packet_id)
    if packet is None:
        raise HTTPException(status_code=404, detail="Violation packet not found.")
    return _packet_to_dict(packet)


@router.post("/violations/{packet_id}/review")
def review_violation(packet_id: str, action: ReviewAction) -> dict[str, Any]:
    if action.packet_id != packet_id:
        raise HTTPException(status_code=400, detail="packet_id mismatch.")
    updated = store.apply_review(action)
    if updated is None:
        raise HTTPException(status_code=404, detail="Violation packet not found.")
    return {
        "message": f"Packet {packet_id} marked as {updated.review_status.value}.",
        "packet": _packet_to_dict(updated),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Analytics
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/analytics")
def get_analytics() -> dict[str, Any]:
    return store.get_analytics()


@router.delete("/violations/reset")
def reset_store() -> dict[str, str]:
    """Dev helper — clear all in-memory violations."""
    store.clear()
    return {"message": "Store cleared."}


# ──────────────────────────────────────────────────────────────────────────────
#  Evidence snapshot serving
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/snapshots/{filename}")
def serve_snapshot(filename: str):
    safe_name = Path(filename).name  # prevent path traversal
    path = SNAPSHOTS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/snapshots/{filename}/base64")
def snapshot_as_base64(filename: str) -> dict[str, str]:
    safe_name = Path(filename).name
    path = SNAPSHOTS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    data = base64.b64encode(path.read_bytes()).decode()
    return {"filename": safe_name, "data": f"data:image/jpeg;base64,{data}"}


# ──────────────────────────────────────────────────────────────────────────────
#  Demo seeder — inject mock violations for UI testing
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/demo/seed")
def seed_demo_violations() -> dict[str, Any]:
    """
    Inject realistic mock violation packets so the dashboard has data
    even without processing a real video. Idempotent — clears first.
    """
    from ..evidence.packet_builder import build_candidate_packet
    import random, time

    store.clear()
    rng = random.Random(42)

    demo_violations = [
        (ViolationType.wrong_side,    0.87, "Zone-B / Indiranagar",    "KA01AB2341"),
        (ViolationType.illegal_parking, 0.91, "Zone-A / MG Road",     "KA03MN5512"),
        (ViolationType.footpath_riding, 0.79, "Zone-C / Koramangala", "KA05PQ7823"),
        (ViolationType.wrong_side,    0.83, "Zone-D / Whitefield",     "TN07RS4490"),
        (ViolationType.illegal_parking, 0.93, "Zone-B / Indiranagar", "KA41CD1234"),
        (ViolationType.footpath_riding, 0.76, "Zone-A / MG Road",     "MH12XY9900"),
        (ViolationType.wrong_side,    0.88, "Zone-C / Koramangala",    "KA02EF3344"),
        (ViolationType.illegal_parking, 0.85, "Zone-D / Whitefield",  "KA50GH7721"),
    ]

    base_time = time.time() - 3600
    for i, (vtype, conf, zone, plate) in enumerate(demo_violations):
        packet = build_candidate_packet(
            violation_type=vtype,
            confidence=conf,
            timestamp_seconds=base_time + i * 420 + rng.uniform(0, 60),
            zone_name=zone,
            evidence_paths=[],
            plate_text=plate,
        )
        store.add(packet)

    return {
        "message": f"Seeded {len(demo_violations)} demo violations.",
        "analytics": store.get_analytics(),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _packet_to_dict(packet) -> dict[str, Any]:
    d = packet.model_dump()
    # Attach snapshot filenames so the UI can fetch them
    d["snapshot_filenames"] = [
        Path(asset["path"]).name
        for asset in d.get("evidence", [])
        if asset.get("kind") == "annotated_frame"
    ]
    return d
