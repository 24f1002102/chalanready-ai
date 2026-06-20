from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detection.detector import create_detector
from .detection.tracker import CentroidTracker, TrackedObject
from .detection.violations.wrong_side import WrongSideRule
from .detection.violations.illegal_parking import IllegalParkingRule
from .detection.violations.footpath_riding import FootpathRidingRule
from .detection.violations.stopline_helmet import StopLineRule, HelmetRule
from .detection.violations.seatbelt_triple import SeatbeltRule, TripleRidingRule
from .evidence.packet_builder import build_candidate_packet
from .models.schemas import ViolationType
from .ocr.plate_reader import PlateReader


@dataclass(frozen=True)
class ProcessingResult:
    input_path: str
    output_path: str
    snapshots_dir: str
    detector: str
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

VIOLATION_COLOR = (0, 50, 255)  # Red for violation overlays


# ─────────────────────────── default scene geometry ──────────────────────────
# These match the synthetic demo video. In production, read from a per-camera
# config JSON so each camera has its own calibrated zones.

_DEFAULT_ROAD_POLYGON = [
    (0, 170), (960, 120), (960, 410), (0, 460),
]
_DEFAULT_FOOTPATH_POLYGON = [
    (0, 72), (960, 42), (960, 122), (0, 170),
]
_DEFAULT_NO_PARK_ZONE = [
    (480, 80), (780, 80), (780, 140), (480, 140),
]
_DEFAULT_ALLOWED_DIRECTION = (1.0, 0.0)   # left-to-right is legal
_DEFAULT_STOP_LINE_Y = 265                 # horizontal dashed line in demo


# ───────────────────────── image preprocessing ───────────────────────────────

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = _CLAHE.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    # Light Gaussian blur to suppress JPEG / compression noise
    denoised = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0.5)
    return denoised


def _make_violation_rules(width: int, height: int):
    """Instantiate all violation rule objects for a given frame size."""
    scale_x = width / 960
    scale_y = height / 540

    def sp(pts):
        return [(int(x * scale_x), int(y * scale_y)) for x, y in pts]

    wrong_side = WrongSideRule(
        allowed_direction=_DEFAULT_ALLOWED_DIRECTION,
        min_points=8,
    )
    parking = IllegalParkingRule(
        restricted_zone=sp(_DEFAULT_NO_PARK_ZONE),
        dwell_frames=45,  # ~2.5 s at 18 fps
    )
    footpath = FootpathRidingRule(
        footpath_zone=sp(_DEFAULT_FOOTPATH_POLYGON),
        min_inside_frames=5,
    )
    stopline = StopLineRule(
        stop_line_y=int(_DEFAULT_STOP_LINE_Y * scale_y),
    )
    helmet = HelmetRule()
    seatbelt = SeatbeltRule()
    triple = TripleRidingRule()
    return wrong_side, parking, footpath, stopline, helmet, seatbelt, triple


# ─────────────────────────────── main pipeline ───────────────────────────────

def process_video(
    input_path: str | Path,
    output_path: str | Path | None = None,
    detector_backend: str = "auto",
    max_frames: int | None = None,
    zone_name: str = "Zone-A / MG Road",
    red_phase_frames: set[int] | None = None,
    store=None,         # optional ViolationsStore to persist packets
) -> dict[str, Any]:

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

    # Directory for violation snapshot JPEGs
    snapshots_dir = output_path.parent / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    tracker = CentroidTracker()
    plate_reader = PlateReader()

    wrong_side, parking, footpath, stopline, helmet, seatbelt, triple = _make_violation_rules(width, height)

    red_phase_frames = red_phase_frames or set()

    frames_processed = 0
    detections_seen = 0
    tracks_seen: set[int] = set()
    violations_detected = 0
    # track_id → set of violation types already flagged (avoid spam)
    already_flagged: dict[int, set[str]] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frames_processed >= max_frames:
                break

            # ── Stage 1: Image Preprocessing (CLAHE + denoise) ───────────────
            preprocessed = preprocess_frame(frame)

            detections = detector.detect(preprocessed)
            tracks = tracker.update(detections, frames_processed)

            detections_seen += len(detections)
            tracks_seen.update(track.track_id for track in tracks)

            violation_flags: list[tuple[int, str, str]] = []  # (track_id, vtype, plate)

            for track in tracks:
                tid = track.track_id
                flagged = already_flagged.setdefault(tid, set())

                # ── Wrong-side driving ─────────────────────────────────────
                if "wrong_side" not in flagged and wrong_side.is_wrong_side(track.centers):
                    flagged.add("wrong_side")
                    plate = plate_reader.read_for_track(tid, preprocessed, track.bbox).text
                    violation_flags.append((tid, "wrong_side", plate))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Wrong-Side Driving", snapshots_dir, frames_processed
                    )
                    if store:
                        packet = build_candidate_packet(
                            violation_type=ViolationType.wrong_side,
                            confidence=0.81,
                            timestamp_seconds=frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate,
                        )
                        store.add(packet)
                    violations_detected += 1

                # ── Illegal parking ────────────────────────────────────────
                if "parking" not in flagged and parking.observe(tid, track.centers):
                    flagged.add("parking")
                    plate = plate_reader.read_for_track(tid, preprocessed, track.bbox).text
                    violation_flags.append((tid, "parking", plate))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Illegal Parking", snapshots_dir, frames_processed
                    )
                    if store:
                        packet = build_candidate_packet(
                            violation_type=ViolationType.illegal_parking,
                            confidence=0.88,
                            timestamp_seconds=frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate,
                        )
                        store.add(packet)
                    violations_detected += 1

                # ── Footpath riding ────────────────────────────────────────
                if "footpath" not in flagged and footpath.observe(tid, track.class_name, track.center):
                    flagged.add("footpath")
                    plate = plate_reader.read_for_track(tid, preprocessed, track.bbox).text
                    violation_flags.append((tid, "footpath", plate))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Footpath Riding", snapshots_dir, frames_processed
                    )
                    if store:
                        packet = build_candidate_packet(
                            violation_type=ViolationType.footpath_riding,
                            confidence=0.85,
                            timestamp_seconds=frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate,
                        )
                        store.add(packet)
                    violations_detected += 1

                # ── Stop-line violation ────────────────────────────────────
                is_red = frames_processed in red_phase_frames
                if "stopline" not in flagged and stopline.observe(
                    tid, track.centers, frames_processed, is_red
                ):
                    flagged.add("stopline")
                    plate = plate_reader.read_for_track(tid, preprocessed, track.bbox).text
                    violation_flags.append((tid, "stopline", plate))
                    snap_path = _save_violation_snapshot(
                        frame, track, "Stop-Line Violation", snapshots_dir, frames_processed
                    )
                    if store:
                        packet = build_candidate_packet(
                            violation_type=ViolationType.stopline,
                            confidence=0.84,
                            timestamp_seconds=frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate,
                        )
                        store.add(packet)
                    violations_detected += 1

                # ── Helmet non-compliance ──────────────────────────
                if "helmet" not in flagged and helmet.observe(
                    tid, track.class_name, track.age, track.bbox, preprocessed
                ):
                    flagged.add("helmet")
                    plate = plate_reader.read_for_track(tid, preprocessed, track.bbox).text
                    violation_flags.append((tid, "helmet", plate))
                    snap_path = _save_violation_snapshot(
                        preprocessed, track, "Helmet Non-Compliance", snapshots_dir, frames_processed
                    )
                    if store:
                        packet = build_candidate_packet(
                            violation_type=ViolationType.helmet,
                            confidence=0.76,
                            timestamp_seconds=frames_processed / fps,
                            zone_name=zone_name,
                            evidence_paths=[snap_path],
                            plate_text=plate,
                        )
                        store.add(packet)
                    violations_detected += 1

                # ── Seatbelt non-compliance (stub — production: YOLOv8-pose) ─
                if "seatbelt" not in flagged and seatbelt.observe(
                    tid, track.class_name, track.age, preprocessed
                ):
                    flagged.add("seatbelt")
                    plate = plate_reader.read_for_track(tid).text
                    violation_flags.append((tid, "seatbelt", plate))
                    _save_violation_snapshot(
                        preprocessed, track, "Seatbelt Non-Compliance", snapshots_dir, frames_processed
                    )
                    violations_detected += 1

                # ── Triple riding (stub — production: multi-person head count) ─
                if "triple" not in flagged and triple.observe(
                    tid, track.class_name, track.age, preprocessed
                ):
                    flagged.add("triple")
                    plate = plate_reader.read_for_track(tid).text
                    violation_flags.append((tid, "triple", plate))
                    _save_violation_snapshot(
                        preprocessed, track, "Triple Riding", snapshots_dir, frames_processed
                    )
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

    result = ProcessingResult(
        input_path=str(input_path),
        output_path=str(output_path),
        snapshots_dir=str(snapshots_dir),
        detector=detector.name,
        frames_processed=frames_processed,
        detections_seen=detections_seen,
        tracks_seen=len(tracks_seen),
        violations_detected=violations_detected,
        fps=float(fps),
        width=width,
        height=height,
    )
    return result.to_dict()


# ──────────────────────────── helper: snapshot ───────────────────────────────

def _save_violation_snapshot(frame, track, label: str, out_dir: Path, frame_idx: int) -> str:
    """Save annotated violation frame as JPEG evidence."""
    annotated = frame.copy()
    x1, y1, x2, y2 = track.bbox
    # Red violation box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), VIOLATION_COLOR, 3)
    # Red banner
    banner_h = 44
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_h), (10, 10, 180), -1)
    cv2.putText(annotated, f"VIOLATION: {label.upper()} | Track #{track.track_id}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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

    # HUD bar
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

    # Violation flash banners (top-right)
    for i, (tid, vtype, plate) in enumerate(violation_flags[:3]):
        label_map = {
            "wrong_side": "WRONG SIDE",
            "parking": "ILLEGAL PARK",
            "footpath": "FOOTPATH RIDE",
            "stopline": "STOP-LINE",
            "helmet": "NO HELMET",
        }
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
    args = parser.parse_args()

    result = process_video(
        input_path=args.input,
        output_path=args.output,
        detector_backend=args.detector,
        max_frames=args.max_frames,
        zone_name=args.zone,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
