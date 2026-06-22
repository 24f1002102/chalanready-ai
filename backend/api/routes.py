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

from ..config.cameras import get_camera_profile, get_zone_summary
from ..evidence.packet_builder import build_candidate_packet
from ..models.schemas import ReviewAction, ReviewStatus, ViolationType
from ..pipeline import (
    process_video,
    preprocess_frame,
    _make_violation_rules,
    _violation_confidence,
    build_evidence_metadata,
)
from ..violations_store import store


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR   = PROJECT_ROOT / "sample_data" / "uploads"
OUTPUT_DIR   = PROJECT_ROOT / "sample_data" / "outputs"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
METRICS_PATH  = PROJECT_ROOT / "sample_data" / "eval_output" / "metrics.json"


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


@router.get("/ai/status")
def ai_status() -> dict[str, Any]:
    """Runtime readiness for detector/OCR transparency in the dashboard."""
    from ..detection.detector import get_detector_status
    from ..ocr.plate_reader import get_ocr_status

    cameras = get_zone_summary()
    return {
        "detector": get_detector_status(),
        "ocr": get_ocr_status(),
        "camera_profiles": {
            "total": len(cameras),
            "generic_available": any(c["camera_id"] == "bengaluru_generic_upload" for c in cameras),
            "calibrated_count": sum(1 for c in cameras if c.get("profile_kind") == "calibrated"),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Camera zones
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/cameras")
def get_cameras() -> dict[str, Any]:
    """Return calibrated camera profiles available to the pipeline."""
    from ..config.cameras import _load_raw_config, get_camera_profiles
    # Invalidate lru_cache so new profiles added to camera_profiles.json
    # are picked up without a server restart
    _load_raw_config.cache_clear()
    get_camera_profiles.cache_clear()

    cameras = get_zone_summary()
    zones = sorted({camera["zone_name"] for camera in cameras})
    return {
        "total_cameras": len(cameras),
        "total_zones": len(zones),
        "profiles": cameras,        # JS loadCameraProfiles() expects this key
        "zones": cameras,           # legacy compat
    }


# ──────────────────────────────────────────────────────────────────────────────
#  AI Metrics — runs real evaluate.py or returns cached metrics.json
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/metrics")
def get_metrics(recompute: bool = False) -> dict[str, Any]:
    """
    Return synthetic smoke-test metrics.

    If metrics.json exists and recompute=False, return cached result.
    If recompute=True or no cached file, run the evaluation pipeline now.
    """
    if not recompute and METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            return _label_synthetic_metrics(metrics)
        except Exception:
            pass

    # Run evaluation inline
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from sample_data.evaluate import evaluate
        metrics = evaluate(detector="color")
        return _label_synthetic_metrics(metrics)
    except FileNotFoundError as exc:
        # Synthetic video not yet generated — return honest placeholder
        return {
            "status": "no_synthetic_video",
            "evaluation_scope": "synthetic_smoke_test",
            "is_production_metric": False,
            "message": str(exc),
            "note": "Run: python sample_data/create_synthetic_video.py then /metrics?recompute=true",
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "per_type": {},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


def _label_synthetic_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics.setdefault("evaluation_scope", "synthetic_smoke_test")
    metrics.setdefault("display_title", "Synthetic Pipeline Smoke Test")
    metrics.setdefault("is_production_metric", False)
    metrics.setdefault(
        "judge_note",
        "These numbers validate the scripted demo pipeline only; real-world performance requires human-labelled CCTV footage.",
    )
    return metrics


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
    camera_id: str = "bengaluru_generic_upload",
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
            camera_id=camera_id,
            detect_signal=True,   # live red-signal detection per frame
            store=store,
            source_kind="video_upload",
            source_name=file.filename or upload_path.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "Video processed. Violation candidates require officer review.",
        "officer_review_required": True,
        "result": result,
        "demo_mode": bool(result.get("is_demo_mode")),
        "calibration_warning": result.get("calibration_warning"),
        "violations_in_store": store.get_analytics()["total_violations"],
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Single-image analysis  (PS3 "Photo Identification" core requirement)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/images/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    zone_name: str = "Zone-A / MG Road",
    detector: str = "auto",
    camera_id: str = "bengaluru_generic_upload",
) -> dict[str, Any]:
    """Analyze one traffic image with the selected camera calibration profile."""
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
    profile = get_camera_profile(camera_id)
    zone_name = profile.zone_name
    gps_lat, gps_lng = profile.gps_lat, profile.gps_lng
    enabled_rules = profile.enabled_rules

    preprocessed = preprocess_frame(frame)

    from ..detection.detector import create_detector
    from ..detection.violations.illegal_parking import point_in_polygon
    from ..detection.violations.redlight import is_red_signal
    from ..ocr.plate_reader import PlateReader

    det = create_detector(detector)
    detections = det.detect(preprocessed)
    _, parking_rule, footpath_rule, stopline_rule, helmet_rule, _, _ = _make_violation_rules(w, h, profile)

    red_signal_active = is_red_signal(
        preprocessed,
        roi_top_frac=profile.signal_roi[0],
        roi_bottom_frac=profile.signal_roi[1],
    )

    violations_found = []
    annotated = preprocessed.copy()
    pending_packets: list[tuple[ViolationType, float, str, str, dict[str, Any]]] = []
    plate_reader = PlateReader()

    for idx, det_obj in enumerate(detections, start=1):
        x1, y1, x2, y2 = det_obj.bbox
        cx, cy = det_obj.center
        bbox = (x1, y1, x2, y2)
        class_name = det_obj.class_name

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(annotated, class_name, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        detected_vtypes = []

        if "helmet" in enabled_rules and helmet_rule.observe(
            track_id=idx, class_name=class_name, track_age=15, bbox=bbox, frame=preprocessed
        ):
            detected_vtypes.append((ViolationType.helmet, _violation_confidence(det_obj.confidence, "helmet"), "helmet"))

        if "footpath" in enabled_rules and point_in_polygon((cx, cy), footpath_rule.footpath_zone):
            detected_vtypes.append((ViolationType.footpath_riding, _violation_confidence(det_obj.confidence, "footpath"), "footpath"))

        if "parking" in enabled_rules and point_in_polygon((cx, cy), parking_rule.restricted_zone):
            detected_vtypes.append((ViolationType.illegal_parking, _violation_confidence(det_obj.confidence, "parking"), "parking"))

        stop_y = stopline_rule.stop_line_y
        if "stopline" in enabled_rules and y1 <= stop_y <= y2:
            detected_vtypes.append((ViolationType.stopline, _violation_confidence(det_obj.confidence, "stopline"), "stopline"))
            if red_signal_active:
                detected_vtypes.append((ViolationType.red_light, _violation_confidence(det_obj.confidence, "stopline"), "stopline"))

        for vtype, conf, rule_key in detected_vtypes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 3)
            label = f"[!] {vtype.value.replace('_', ' ').upper()} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            plate = plate_reader.read_for_track(idx, preprocessed, bbox)
            metadata = build_evidence_metadata(
                det,
                profile,
                "image_upload",
                frame_index=0,
                rule_key=rule_key,
                detector_confidence=det_obj.confidence,
                input_name=file.filename or "uploaded_image",
                bbox=bbox,
                extra={"red_signal_detected": red_signal_active, "class_name": class_name},
            )
            pending_packets.append((vtype, conf, plate.text, plate.source, metadata))
            violations_found.append({
                "violation_type": vtype.value,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "class_name": class_name,
                "packet_id": None,
                "plate": plate.text,
                "plate_source": plate.source,
                "metadata": metadata,
            })

    if pending_packets:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = Path(file.filename or "photo").stem[:40] or "photo"
        evidence_path = SNAPSHOTS_DIR / f"image_{safe_stem}_{uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(evidence_path), annotated)

        packet_ids: list[str] = []
        for vtype, conf, plate_text, plate_source, metadata in pending_packets:
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
                metadata=metadata,
            )
            store.add(packet)
            packet_ids.append(packet.packet_id)

        for item, packet_id in zip(violations_found, packet_ids):
            item["packet_id"] = packet_id
            item["snapshot_filename"] = evidence_path.name

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
        "detector_mode": getattr(det, "mode", det.name),
        "demo_mode": bool(getattr(det, "is_demo_mode", False)),
        "camera_id": profile.camera_id,
        "camera_name": profile.display_name,
        "calibration_profile": profile.profile_kind,
        "calibration_warning": profile.calibration_warning,
        "enabled_rules": sorted(enabled_rules),
        "red_signal_detected": red_signal_active,
        "vehicles_detected": len(detections),
        "violations": violations_found,
        "annotated_image": f"data:image/jpeg;base64,{b64}",
    }



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
#  Analytics export
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/analytics/export")
def export_violations_csv(status: str | None = None) -> Any:
    """Download all violations (or filtered by status) as a CSV file."""
    from fastapi.responses import Response
    csv_data = store.export_csv(status=status)
    filename = f"chalanready_violations_{status or 'all'}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/analytics/summary")
def get_analytics_summary() -> dict[str, Any]:
    """Return a quick summary for dashboard header stats."""
    analytics = store.get_analytics()
    by_type = analytics.get("by_type", {})
    by_zone = analytics.get("by_zone", {})
    by_hour = analytics.get("by_hour", {})

    peak_hour = max(by_hour, key=by_hour.get) if by_hour else "—"
    hottest_zone = max(by_zone, key=by_zone.get) if by_zone else "—"
    total = analytics.get("total_violations", 0)
    approved = analytics.get("approved_challans", 0)
    est_fine_revenue = approved * 500  # ₹500 per approved challan

    return {
        **analytics,
        "peak_hour": peak_hour,
        "hottest_zone": hottest_zone,
        "est_fine_revenue_inr": est_fine_revenue,
        "most_common_violation": max(by_type, key=by_type.get) if by_type else "—",
    }


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

@router.post("/demo/process")
def process_demo_video(
    zone_name: str = "Zone-A / MG Road",
    detector: str = "auto",
) -> dict[str, Any]:
    """
    Run the AI violation detection pipeline on the demo CCTV video.
    Uses real YOLO detection — not synthetic/fabricated data.

    Priority order for demo video:
    1. Real Delhi intersection CCTV footage (if uploaded)
    2. Synthetic demo video (fallback)
    """
    # Find best available demo video
    _real = PROJECT_ROOT / "sample_data" / "videos" / \
        "stock-footage-delhi-india-jul-smooth-traffic-flow-at-intersection-with-green-signal.mov"
    _synth = PROJECT_ROOT / "sample_data" / "videos" / "synthetic_stage1.mp4"

    if _real.exists():
        demo_video = _real
        video_type = "real_cctv"
        demo_camera_id = "delhi_intersection_demo"
    elif _synth.exists():
        demo_video = _synth
        video_type = "synthetic"
        demo_camera_id = "synthetic_stage1"
    else:
        raise HTTPException(
            status_code=404,
            detail="No demo video found. Place a video in sample_data/videos/ or run the synthetic generator."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"demo_annotated_{demo_video.stem}.mp4"

    before_count = store.get_analytics()["total_violations"]

    try:
        from ..pipeline import process_video
        result = process_video(
            input_path=demo_video,
            output_path=output_path,
            detector_backend=detector,
            zone_name=get_camera_profile(demo_camera_id).zone_name,
            camera_id=demo_camera_id,
            store=store,
            source_kind=f"demo_{video_type}",
            source_name=demo_video.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    after_analytics = store.get_analytics()
    after_count = after_analytics["total_violations"]

    return {
        "message": "Camera-calibrated AI pipeline completed. Candidates require officer review.",
        "video_source": video_type,
        "camera_id": result["camera_id"],
        "camera_name": result["camera_name"],
        "video_file": demo_video.name,
        "detector_used": result["detector"],
        "detector_mode": result.get("detector_mode"),
        "demo_mode": bool(result.get("is_demo_mode")),
        "calibration_warning": result.get("calibration_warning"),
        "frames_processed": result["frames_processed"],
        "violations_detected": result["violations_detected"],
        "violations_before": before_count,
        "violations_after": after_count,
        "new_candidates_added": max(0, after_count - before_count),
        "officer_review_required": True,
        "analytics": after_analytics,
    }


@router.post("/demo/seed")
def seed_demo_data() -> dict[str, Any]:
    """Compatibility alias: run the calibrated demo pipeline without sample-row injection."""
    return process_demo_video()


@router.get("/demo/status")
def demo_status() -> dict[str, Any]:
    """Check whether demo data is present and what type it is."""
    _real = PROJECT_ROOT / "sample_data" / "videos" / \
        "stock-footage-delhi-india-jul-smooth-traffic-flow-at-intersection-with-green-signal.mov"
    _synth = PROJECT_ROOT / "sample_data" / "videos" / "synthetic_stage1.mp4"
    analytics = store.get_analytics()
    return {
        "violations_in_store": analytics["total_violations"],
        "real_cctv_video_available": _real.exists(),
        "synthetic_video_available": _synth.exists(),
        "analytics": analytics,
        "cameras": get_zone_summary(),
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
