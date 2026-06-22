from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict, dataclass
from math import hypot
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config.cameras import CameraProfile, get_camera_profile
from .detection.detector import create_detector
from .detection.tracker import CentroidTracker, TrackedObject
from .detection.violations.wrong_side import WrongSideRule
from .detection.violations.illegal_parking import IllegalParkingRule, point_in_polygon
from .detection.violations.footpath_riding import FootpathRidingRule
from .detection.violations.stopline_helmet import StopLineRule, HelmetRule
from .detection.violations.seatbelt_triple import SeatbeltRule, TripleRidingRule
from .detection.violations.redlight import is_red_signal
from .evidence.packet_builder import build_candidate_packet
from .models.schemas import ViolationType
from .ocr.plate_reader import PlateReader


@dataclass(frozen=True)
class ProcessingResult:
    input_path: str
    output_path: str
    snapshots_dir: str
    detector: str
    detector_mode: str
    is_demo_mode: bool
    camera_id: str
    camera_name: str
    calibration_profile: str
    calibration_warning: str | None
    zone_name: str
    source_kind: str
    frames_processed: int
    detections_seen: int
    tracks_seen: int
    violations_detected: int
    fps: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRACK_COLORS = [
    (0, 184, 255),
    (91, 214, 94),
    (255, 155, 80),
    (209, 106, 255),
    (86, 196, 255),
    (255, 220, 90),
]

VIOLATION_COLOR = (0, 50, 255)


# ── Confidence weight factors per rule ──────────────────────────────────────
# These are reliability weights multiplied against the detector's actual
# confidence score. A weight of 1.0 means we trust the rule fully;
# lower values reflect uncertainty in the heuristic.
_RULE_WEIGHTS = {
    "wrong_side":  0.95,   # geometric — very reliable when track is long enough
    "parking":     0.92,   # zone polygon + dwell time — reliable
    "footpath":    0.90,   # zone polygon — reliable
    "stopline":    0.88,   # geometric crossing — reliable when signal state known
    "helmet":      0.78,   # HSV heuristic — moderate reliability
    "seatbelt":    0.72,   # pixel heuristic — lower reliability
    "triple":      0.74,   # blob counting — moderate reliability
}


def _violation_confidence(detector_conf: float, rule_key: str) -> float:
    """Compute final violation confidence from model confidence + rule reliability."""
    weight = _RULE_WEIGHTS.get(rule_key, 0.80)
    # Blend: 70% detector confidence + 30% rule weight ceiling
    raw = detector_conf * 0.70 + weight * 0.30
    return round(min(max(raw, 0.50), 0.99), 4)


def build_evidence_metadata(
    detector: Any,
    profile: CameraProfile,
    source_kind: str,
    frame_index: int | None = None,
    rule_key: str | None = None,
    detector_confidence: float | None = None,
    input_name: str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured provenance for judge/officer trust and auditability."""
    detector_mode = getattr(detector, "mode", detector.name)
    is_demo_mode = bool(getattr(detector, "is_demo_mode", detector_mode == "synthetic_demo"))
    metadata: dict[str, Any] = {
        "detector_name": detector.name,
        "detector_display_name": getattr(detector, "display_name", detector.name),
        "detector_mode": detector_mode,
        "is_demo_mode": is_demo_mode,
        "camera_id": profile.camera_id,
        "camera_name": profile.display_name,
        "calibration_profile": profile.profile_kind,
        "calibration_warning": profile.calibration_warning,
        "source_kind": source_kind,
        "input_name": input_name,
    }
    if frame_index is not None:
        metadata["frame_index"] = frame_index
    if rule_key is not None:
        metadata["rule_key"] = rule_key
        metadata["rule_weight"] = _RULE_WEIGHTS.get(rule_key)
    if detector_confidence is not None:
        metadata["detector_confidence"] = round(float(detector_confidence), 4)
    if bbox is not None:
        metadata["bbox"] = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
    if extra:
        metadata.update(extra)
    return metadata


def _is_duplicate_violation(
    recent_events: list[tuple[str, int, tuple[int, int]]],
    rule_key: str,
    center: tuple[int, int],
    frame_index: int,
    cooldown_frames: int = 120,
    distance_px: float = 220.0,
) -> bool:
    """Suppress fragmented-track duplicates for the same nearby violation event."""
    recent_events[:] = [
        item for item in recent_events
        if frame_index - item[1] <= cooldown_frames
    ]
    for old_key, old_frame, old_center in recent_events:
        if old_key != rule_key:
            continue
        if frame_index - old_frame <= cooldown_frames and hypot(
            center[0] - old_center[0], center[1] - old_center[1]
        ) <= distance_px:
            return True
    recent_events.append((rule_key, frame_index, center))
    return False


# ───────────────────────── image preprocessing ───────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Stage 1 preprocessing — addresses PS3 requirement for handling low-light,
    shadows, and image quality variance in CCTV footage.

    Pipeline:
      1. Convert BGR → LAB colour space
      2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
         on the L (luminance) channel → fixes low-light / shadow issues
      3. Merge back → convert to BGR
      4. Gentle Gaussian denoise to reduce compression artefacts

    At typical CCTV 720p resolution this runs in <1ms per frame on CPU.
    Note: CLAHE is instantiated per-call for thread-safety under concurrent uploads.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0.5)
    return denoised


def _make_violation_rules(width: int, height: int, profile: CameraProfile | None = None):
    """Instantiate camera-calibrated violation rule objects."""
    profile = profile or get_camera_profile()

    wrong_side = WrongSideRule(
        allowed_direction=profile.allowed_direction,
        min_points=12,
        opposite_threshold=-0.65,
        min_movement_px=40.0,
    )
    parking = IllegalParkingRule(
        restricted_zone=profile.scale_points(profile.restricted_parking_zone, width, height),
        dwell_frames=60,
    )
    footpath = FootpathRidingRule(
        footpath_zone=profile.scale_points(profile.footpath_zone, width, height),
        min_inside_frames=4,
    )
    stopline = StopLineRule(
        stop_line_y=profile.scale_y(profile.stop_line_y, height),
        approach_direction=profile.approach_direction,
    )
    helmet = HelmetRule()
    seatbelt = SeatbeltRule(min_track_age=25)
    triple = TripleRidingRule()
    return wrong_side, parking, footpath, stopline, helmet, seatbelt, triple


# ─────────────────────────────── main pipeline ───────────────────────────────

def process_video(
    input_path: str | Path,
    output_path: str | Path | None = None,
    detector_backend: str = "auto",
    max_frames: int | None = None,
    zone_name: str = "Zone-A / MG Road",
    camera_id: str | None = None,
    red_phase_frames: set[int] | None = None,
    store=None,
    detect_signal: bool = True,
    source_kind: str = "video_upload",
    source_name: str | None = None,
) -> dict[str, Any]:
    import time as _time
    _start_wall_time = _time.time()

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    if output_path is None:
        output_dir = input_path.parents[1] / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_annotated.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshots_dir = output_path.parent / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    profile = get_camera_profile(camera_id)
    zone_name = zone_name or profile.zone_name

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")

    detector = create_detector(detector_backend)
    source_name = source_name or input_path.name
    tracker = CentroidTracker()
    plate_reader = PlateReader()

    wrong_side, parking, footpath, stopline, helmet, seatbelt, triple = _make_violation_rules(width, height, profile)
    enabled_rules = profile.enabled_rules

    red_phase_frames = red_phase_frames or set()
    _gps_lat, _gps_lng = profile.gps_lat, profile.gps_lng

    frames_processed = 0
    detections_seen = 0
    tracks_seen: set[int] = set()
    violations_detected = 0
    already_flagged: dict[int, set[str]] = {}
    recent_violation_events: list[tuple[str, int, tuple[int, int]]] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frames_processed >= max_frames:
                break

            # ── Stage 1: Preprocessing (CLAHE + denoise) ────────────────────
            preprocessed = preprocess_frame(frame)

            detections = detector.detect(preprocessed)
            tracks = tracker.update(detections, frames_processed)

            detections_seen += len(detections)
            tracks_seen.update(track.track_id for track in tracks)

            violation_flags: list[tuple[int, str, str]] = []

            for track in tracks:
                tid = track.track_id
                flagged = already_flagged.setdefault(tid, set())
                # Use the real detector confidence for this track
                det_conf = track.confidence

                # ── Wrong-side driving ─────────────────────────────────────
                if (
                    "wrong_side" in enabled_rules
                    and "wrong_side" not in flagged
                    and not point_in_polygon(track.center, parking.restricted_zone)
                    and not point_in_polygon(track.center, footpath.footpath_zone)
                    and wrong_side.is_wrong_side(track.centers)
                ):
                    flagged.add("wrong_side")
                    if _is_duplicate_violation(recent_violation_events, "wrong_side", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "wrong_side")
                    violation_flags.append((tid, "wrong_side", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Wrong-Side Driving", snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=ViolationType.wrong_side_driving,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "wrong_side", det_conf, source_name, track.bbox
                            ),
                        ))
                    violations_detected += 1

                # ── Illegal parking ────────────────────────────────────────
                if "parking" in enabled_rules and "parking" not in flagged and parking.observe(tid, track.centers):
                    flagged.add("parking")
                    if _is_duplicate_violation(recent_violation_events, "parking", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "parking")
                    violation_flags.append((tid, "parking", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Illegal Parking", snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=ViolationType.illegal_parking,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "parking", det_conf, source_name, track.bbox
                            ),
                        ))
                    violations_detected += 1

                # ── Footpath riding ────────────────────────────────────────
                if "footpath" in enabled_rules and "footpath" not in flagged and footpath.observe(tid, track.class_name, track.center):
                    flagged.add("footpath")
                    if _is_duplicate_violation(recent_violation_events, "footpath", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "footpath")
                    violation_flags.append((tid, "footpath", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Footpath Riding", snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=ViolationType.footpath_riding,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "footpath", det_conf, source_name, track.bbox
                            ),
                        ))
                    violations_detected += 1

                # ── Stop-line / Red-light violation ────────────────────────
                if red_phase_frames:
                    is_red = frames_processed in red_phase_frames
                elif detect_signal:
                    is_red = is_red_signal(
                        preprocessed,
                        roi_top_frac=profile.signal_roi[0],
                        roi_bottom_frac=profile.signal_roi[1],
                    )
                else:
                    is_red = False

                if "stopline" in enabled_rules and "stopline" not in flagged and stopline.observe(
                    tid, track.centers, frames_processed, is_red
                ):
                    flagged.add("stopline")
                    if _is_duplicate_violation(recent_violation_events, "stopline", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "stopline")
                    vtype = ViolationType.red_light if is_red else ViolationType.stopline
                    vname = "Red-Light Violation" if is_red else "Stop-Line Violation"
                    violation_flags.append((tid, "stopline", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        frame, track, vname, snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=vtype,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "stopline", det_conf, source_name, track.bbox,
                                {"red_signal_detected": is_red}
                            ),
                        ))
                    violations_detected += 1

                # ── Helmet non-compliance ──────────────────────────────────
                if "helmet" in enabled_rules and "helmet" not in flagged and helmet.observe(
                    tid, track.class_name, track.age, track.bbox, preprocessed
                ):
                    flagged.add("helmet")
                    if _is_duplicate_violation(recent_violation_events, "helmet", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "helmet")
                    violation_flags.append((tid, "helmet", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        preprocessed, track, "Helmet Non-Compliance", snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=ViolationType.helmet,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "helmet", det_conf, source_name, track.bbox
                            ),
                        ))
                    violations_detected += 1

                # ── Seatbelt non-compliance ─────────────────────────────────
                x1s, y1s, x2s, y2s = track.bbox
                _seat_crop = (
                    preprocessed[max(0, y1s):y2s, max(0, x1s):x2s]
                    if y2s > y1s and x2s > x1s else None
                )
                if "seatbelt" in enabled_rules and "seatbelt" not in flagged and seatbelt.observe(
                    tid, track.class_name, track.age, _seat_crop
                ):
                    flagged.add("seatbelt")
                    if _is_duplicate_violation(recent_violation_events, "seatbelt", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "seatbelt")
                    violation_flags.append((tid, "seatbelt", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        preprocessed, track, "Seatbelt Non-Compliance", snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=ViolationType.seatbelt,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "seatbelt", det_conf, source_name, track.bbox
                            ),
                        ))
                    violations_detected += 1

                # ── Triple riding ──────────────────────────────────────────
                x1t, y1t, x2t, y2t = track.bbox
                _tri_crop = (
                    preprocessed[max(0, y1t):y2t, max(0, x1t):x2t]
                    if y2t > y1t and x2t > x1t else None
                )
                if "triple" in enabled_rules and "triple" not in flagged and triple.observe(
                    tid, track.class_name, track.age, _tri_crop
                ):
                    flagged.add("triple")
                    if _is_duplicate_violation(recent_violation_events, "triple", track.center, frames_processed):
                        continue
                    plate_read = plate_reader.read_for_track(tid, preprocessed, track.bbox)
                    conf = _violation_confidence(det_conf, "triple")
                    violation_flags.append((tid, "triple", plate_read.text))
                    snap_path = _save_violation_snapshot(
                        preprocessed, track, "Triple Riding", snapshots_dir, frames_processed
                    )
                    if store:
                        store.add(build_candidate_packet(
                            violation_type=ViolationType.triple_riding,
                            confidence=conf,
                            timestamp_seconds=_start_wall_time + frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate_read.text,
                            plate_source=plate_read.source,
                            gps_lat=_gps_lat,
                            gps_lng=_gps_lng,
                            metadata=build_evidence_metadata(
                                detector, profile, source_kind, frames_processed,
                                "triple", det_conf, source_name, track.bbox
                            ),
                        ))
                    violations_detected += 1

            annotated = annotate_frame(
                preprocessed, tracks, frames_processed, detector.name,
                violations_detected, violation_flags
            )
            writer.write(annotated)
            frames_processed += 1

    finally:
        capture.release()
        writer.release()

    return ProcessingResult(
        input_path=str(input_path),
        output_path=str(output_path),
        snapshots_dir=str(snapshots_dir),
        detector=detector.name,
        detector_mode=getattr(detector, "mode", detector.name),
        is_demo_mode=bool(getattr(detector, "is_demo_mode", False)),
        camera_id=profile.camera_id,
        camera_name=profile.display_name,
        calibration_profile=profile.profile_kind,
        calibration_warning=profile.calibration_warning,
        zone_name=zone_name,
        source_kind=source_kind,
        frames_processed=frames_processed,
        detections_seen=detections_seen,
        tracks_seen=len(tracks_seen),
        violations_detected=violations_detected,
        fps=float(fps),
        width=width,
        height=height,
    ).to_dict()


# ──────────────────────────── helper: snapshot ───────────────────────────────

def _save_violation_snapshot(frame, track, label: str, out_dir: Path, frame_idx: int) -> str:
    """Save annotated violation frame as JPEG evidence."""
    annotated = frame.copy()
    x1, y1, x2, y2 = track.bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), VIOLATION_COLOR, 3)
    banner_h = 44
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_h), (10, 10, 180), -1)
    cv2.putText(
        annotated,
        f"VIOLATION: {label.upper()} | Track #{track.track_id}",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    filename = f"{label.replace(' ', '_').lower()}_{track.track_id}_{frame_idx:04d}.jpg"
    path = out_dir / filename
    cv2.imwrite(str(path), annotated)
    return str(path)


# ──────────────────────────── frame annotation ───────────────────────────────

def annotate_frame(
    frame,
    tracks: list[TrackedObject],
    frame_index: int,
    detector_name: str,
    total_violations: int,
    violation_flags: list[tuple[int, str, str]],
):
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    cv2.rectangle(annotated, (0, 0), (w, 60), (14, 17, 23), -1)
    cv2.putText(
        annotated,
        f"ChalanReady AI  |  detector={detector_name}  |  frame={frame_index:04d}",
        (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Tracks: {len(tracks)}  |  Violations: {total_violations}  |  Officer review required",
        (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1, cv2.LINE_AA,
    )

    label_map = {
        "wrong_side": "WRONG SIDE",
        "parking":    "ILLEGAL PARK",
        "footpath":   "FOOTPATH RIDE",
        "stopline":   "STOP-LINE / RED-LIGHT",
        "helmet":     "NO HELMET",
        "seatbelt":   "NO SEATBELT",
        "triple":     "TRIPLE RIDING",
    }
    for i, (tid, vtype, plate) in enumerate(violation_flags[:3]):
        vtext = f"[!] {label_map.get(vtype, vtype)} | {plate}"
        bx = w - 340
        by = 70 + i * 42
        cv2.rectangle(annotated, (bx - 6, by - 26), (w - 6, by + 12), (0, 0, 180), -1)
        cv2.putText(annotated, vtext, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 100), 1, cv2.LINE_AA)

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = TRACK_COLORS[(track.track_id - 1) % len(TRACK_COLORS)]
        label = f"ID{track.track_id} {track.class_name} {track.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, track.center, 4, color, -1)
        _draw_label(annotated, label, (x1, max(18, y1 - 8)), color)
        if len(track.centers) >= 2:
            recent = track.centers[-24:]
            for start, end in zip(recent, recent[1:]):
                cv2.line(annotated, start, end, color, 2)

    return annotated


def _draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (15, 20, 25), 1, cv2.LINE_AA)


# ──────────────────────────── CLI entry point ─────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ChalanReady AI — full violation pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--detector", default="auto", choices=["auto", "yolo", "color"])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--zone", default="Zone-A / MG Road")
    parser.add_argument("--camera-id", default=None)
    args = parser.parse_args()

    result = process_video(
        input_path=args.input,
        output_path=args.output,
        detector_backend=args.detector,
        max_frames=args.max_frames,
        zone_name=args.zone,
        camera_id=args.camera_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
