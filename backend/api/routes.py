from __future__ import annotations

import base64
import json
import os
import time
import uuid as _uuid
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ..evidence.packet_builder import build_candidate_packet
from ..models.schemas import ReviewAction, ReviewStatus, ViolationType
from ..pipeline import process_video, preprocess_frame, _make_violation_rules
from ..violations_store import store


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR   = PROJECT_ROOT / "sample_data" / "uploads"
OUTPUT_DIR   = PROJECT_ROOT / "sample_data" / "outputs"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
METRICS_PATH  = PROJECT_ROOT / "sample_data" / "eval_output" / "metrics.json"

# ── Zone registry — real Bengaluru GPS coords per enforcement zone ────────────
ZONE_REGISTRY: list[dict[str, Any]] = [
    {"name": "Zone-A / MG Road",       "lat": 12.9716, "lng": 77.5946, "cameras": 4},
    {"name": "Zone-B / Indiranagar",   "lat": 12.9784, "lng": 77.6408, "cameras": 3},
    {"name": "Zone-C / Koramangala",   "lat": 12.9352, "lng": 77.6245, "cameras": 3},
    {"name": "Zone-D / Whitefield",    "lat": 12.9698, "lng": 77.7500, "cameras": 2},
    {"name": "Zone-E / Jayanagar",     "lat": 12.9250, "lng": 77.5938, "cameras": 2},
]

def _zone_gps(zone_name: str) -> tuple[float, float]:
    """Return (lat, lng) for a zone name, falling back to city centre."""
    for z in ZONE_REGISTRY:
        if z["name"] == zone_name:
            return float(z["lat"]), float(z["lng"])
    return 12.9716, 77.5946


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
#  Camera zones
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/cameras")
def get_cameras() -> dict[str, Any]:
    """Return the list of active enforcement camera zones."""
    total = sum(z["cameras"] for z in ZONE_REGISTRY)
    return {
        "total_cameras": total,
        "total_zones": len(ZONE_REGISTRY),
        "zones": ZONE_REGISTRY,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  AI Metrics — runs real evaluate.py or returns cached metrics.json
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/metrics")
def get_metrics(recompute: bool = False) -> dict[str, Any]:
    """
    Return real AI evaluation metrics.

    If metrics.json exists and recompute=False, return cached result.
    If recompute=True or no cached file, run the evaluation pipeline now.
    """
    if not recompute and METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Run evaluation inline
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from sample_data.evaluate import evaluate
        metrics = evaluate(detector="color")
        return metrics
    except FileNotFoundError as exc:
        # Synthetic video not yet generated — return honest placeholder
        return {
            "status": "no_synthetic_video",
            "message": str(exc),
            "note": "Run: python sample_data/create_synthetic_video.py then /metrics?recompute=true",
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "per_type": {},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


@router.post("/metrics/generate-video")
def generate_synthetic_video() -> dict[str, str]:
    """Generate the synthetic test video if it doesn't exist yet."""
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from sample_data.create_synthetic_video import main as gen_main
        gen_main()
        return {"message": "Synthetic video generated successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
            detect_signal=True,   # live red-signal detection per frame
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
#  Single-image analysis  (PS3 "Photo Identification" core requirement)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/images/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    zone_name: str = "Zone-A / MG Road",
    detector: str = "auto",     # default to auto (YOLOv8 → color fallback)
) -> dict[str, Any]:
    """
    Analyze a single traffic photo for violations.

    Supported static violations (single-frame detectable):
      - Helmet non-compliance  (HSV skin-tone heuristic on motorcycle bbox)
      - Illegal parking         (vehicle center inside no-park polygon)
      - Footpath riding         (vehicle center inside footpath polygon)
      - Stop-line violation     (vehicle bbox crosses stop-line Y)
      - Red-light violation     (bright red signal region detected above stop-line)
    """
    suffix = Path(file.filename or "photo.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Upload an image file (.jpg / .jpeg / .png / .bmp / .webp).",
        )

    raw_bytes = await file.read()
    img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid photo.")

    h, w = frame.shape[:2]
    gps_lat, gps_lng = _zone_gps(zone_name)

    # Stage 1 — CLAHE preprocessing
    preprocessed = preprocess_frame(frame)

    # Stage 2 — Vehicle detection
    from ..detection.detector import create_detector
    det = create_detector(detector)
    detections = det.detect(preprocessed)

    # Stage 3 — Static violation checks
    _, parking_rule, footpath_rule, stopline_rule, helmet_rule, _, _ = _make_violation_rules(w, h)

    from ..detection.violations.illegal_parking import point_in_polygon
    from ..detection.violations.redlight import is_red_signal
    from ..ocr.plate_reader import PlateReader

    violations_found = []
    annotated = preprocessed.copy()
    pending_packets: list[tuple[ViolationType, float, str, str]] = []
    plate_reader = PlateReader()

    # Check signal state for red-light
    red_signal_active = is_red_signal(preprocessed)

    for idx, det_obj in enumerate(detections, start=1):
        x1, y1, x2, y2 = det_obj.bbox
        cx, cy = det_obj.center
        bbox = (x1, y1, x2, y2)
        class_name = det_obj.class_name

        # Draw vehicle box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(annotated, class_name, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        detected_vtypes = []

        # Helmet check
        if helmet_rule.observe(
            track_id=idx, class_name=class_name, track_age=15,
            bbox=bbox, frame=preprocessed
        ):
            detected_vtypes.append((ViolationType.helmet, 0.72))

        # Footpath check
        if point_in_polygon((cx, cy), footpath_rule.footpath_zone):
            detected_vtypes.append((ViolationType.footpath_riding, 0.81))

        # Parking-zone check
        if point_in_polygon((cx, cy), parking_rule.restricted_zone):
            detected_vtypes.append((ViolationType.illegal_parking, 0.85))

        # Stop-line check
        stop_y = stopline_rule.stop_line_y
        if y1 <= stop_y <= y2:
            detected_vtypes.append((ViolationType.stopline, 0.78))
            if red_signal_active:
                detected_vtypes.append((ViolationType.red_light, 0.82))

        for vtype, conf in detected_vtypes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 3)
            label = f"[!] {vtype.value.replace('_', ' ').upper()} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            plate = plate_reader.read_for_track(idx, preprocessed, bbox)
            pending_packets.append((vtype, conf, plate.text, plate.source))
            violations_found.append({
                "violation_type": vtype.value,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "class_name": class_name,
                "packet_id": None,
                "plate": plate.text,
                "plate_source": plate.source,
            })

    if pending_packets:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = Path(file.filename or "photo").stem[:40] or "photo"
        evidence_path = SNAPSHOTS_DIR / f"image_{safe_stem}_{uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(evidence_path), annotated)

        packet_ids: list[str] = []
        for vtype, conf, plate_text, plate_source in pending_packets:
            gps_lat, gps_lng = _zone_gps(zone_name)
            packet = build_candidate_packet(
                violation_type=vtype,
                confidence=conf,
                timestamp_seconds=time.time(),
                zone_name=zone_name,
                evidence_paths=[str(evidence_path)],
                plate_text=plate_text,
                plate_source=plate_source,
                gps_lat=gps_lat,
                gps_lng=gps_lng,
            )
            store.add(packet)
            packet_ids.append(packet.packet_id)

        for item, packet_id in zip(violations_found, packet_ids):
            item["packet_id"] = packet_id
            item["snapshot_filename"] = evidence_path.name

    # Encode annotated image to base64
    success, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "message": f"Image analyzed. {len(violations_found)} violation(s) detected. Officer review required.",
        "officer_review_required": True,
        "image_size": {"width": w, "height": h},
        "preprocessing": "CLAHE + Gaussian denoise applied",
        "detector_used": det.name,
        "red_signal_detected": red_signal_active,
        "vehicles_detected": len(detections),
        "violations": violations_found,
        "annotated_image": f"data:image/jpeg;base64,{b64}",
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
    """Dev helper — clear all violations."""
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
#  Demo seeder — inject demo violations for UI testing (clearly labelled)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/demo/seed")
def seed_demo_violations() -> dict[str, Any]:
    """
    Inject demo violation packets so the dashboard has data
    without processing a real video. All plates are DEMO-labelled.
    Idempotent — clears first.
    """
    import random

    store.clear()
    rng = random.Random(42)

    demo_violations = [
        (ViolationType.wrong_side_driving,  0.87, "Zone-B / Indiranagar"),
        (ViolationType.illegal_parking,     0.91, "Zone-A / MG Road"),
        (ViolationType.footpath_riding,     0.79, "Zone-C / Koramangala"),
        (ViolationType.wrong_side_driving,  0.83, "Zone-D / Whitefield"),
        (ViolationType.illegal_parking,     0.93, "Zone-B / Indiranagar"),
        (ViolationType.footpath_riding,     0.76, "Zone-A / MG Road"),
        (ViolationType.wrong_side_driving,  0.88, "Zone-C / Koramangala"),
        (ViolationType.illegal_parking,     0.85, "Zone-D / Whitefield"),
    ]

    # Generate realistic-looking demo plates (clearly marked as demo)
    _STATE_CODES = ["KA", "KA", "KA", "KA", "TN", "AP", "MH"]
    _DISTRICTS   = ["01", "02", "03", "04", "05", "41", "50"]
    _LETTERS     = "ABCDEFGHJKLMNPQRSTUVWXYZ"

    def _demo_plate(seed_val: int) -> str:
        r = random.Random(seed_val)
        return (
            r.choice(_STATE_CODES)
            + r.choice(_DISTRICTS)
            + "".join(r.choices(_LETTERS, k=2))
            + str(r.randint(1000, 9999))
        )

    base_time = time.time() - 3600
    for i, (vtype, conf, zone) in enumerate(demo_violations):
        gps_lat, gps_lng = _zone_gps(zone)
        plate = _demo_plate(i * 17 + 3)
        packet = build_candidate_packet(
            violation_type=vtype,
            confidence=conf,
            timestamp_seconds=base_time + i * 420 + rng.uniform(0, 60),
            zone_name=zone,
            evidence_paths=[],
            plate_text=plate,
            plate_source="synthetic_demo_seed",
            gps_lat=gps_lat,
            gps_lng=gps_lng,
        )
        store.add(packet)

    return {
        "message": f"Seeded {len(demo_violations)} demo violations. All data is synthetic — not real detection output.",
        "is_demo_data": True,
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
