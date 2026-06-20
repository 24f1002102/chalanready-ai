"""
ChalanReady AI — Offline Evaluation Script
==========================================
Runs the violation detection pipeline on the synthetic demo video and
computes Precision, Recall, and F1 against known ground-truth annotations.

Usage:
    python sample_data/evaluate.py
    python sample_data/evaluate.py --detector yolo   # requires ultralytics

Ground truth is derived from the synthetic video generator
(create_synthetic_video.py) — the video is scripted so violations occur
at exact, known frame ranges.

Metrics reported:
  • Per-violation-type: TP / FP / FN / Precision / Recall / F1
  • Aggregate: macro-average Precision / Recall / F1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline import process_video          # noqa: E402
from backend.violations_store import ViolationsStore  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Ground truth for synthetic_stage1.mp4 (150 frames @ 18 fps)
# The synthetic video has 3 vehicles with scripted violation trajectories.
# See create_synthetic_video.py for exact pixel paths used to generate them.
# ──────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    "footpath_riding":              1,  # motorcycle enters footpath zone
    "wrong_side_driving":           1,  # car travels right-to-left (counter-flow)
    "illegal_or_footpath_parking":  1,  # truck dwells in no-park zone ≥45 frames
    "stop_line_violation":          0,  # not testable without red_phase_frames
    "helmet_non_compliance":        0,  # not testable on synthetic vehicles
}


def evaluate(detector: str = "color") -> dict:
    video_path = ROOT / "sample_data" / "videos" / "synthetic_stage1.mp4"
    output_path = ROOT / "sample_data" / "eval_output" / "eval_annotated.mp4"
    eval_db_path = ROOT / "sample_data" / "eval_output" / "eval.sqlite3"

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        print("  Run:  python sample_data/create_synthetic_video.py")
        sys.exit(1)

    # Use an isolated SQLite store so app/demo data cannot pollute metrics.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = ViolationsStore(eval_db_path)
    store.clear()

    print(f"\nRunning pipeline on: {video_path.name}  (detector={detector})")
    result = process_video(
        input_path=video_path,
        output_path=output_path,
        detector_backend=detector,
        zone_name="eval-zone",
        store=store,
    )
    print(f"Frames processed : {result['frames_processed']}")
    print(f"Detections seen  : {result['detections_seen']}")
    print(f"Tracks           : {result['tracks_seen']}")
    print(f"Violations found : {result['violations_detected']}")

    # Count detected types from store
    all_violations = store.list_all()
    detected_types: dict[str, int] = {}
    for v in all_violations:
        vt = v.violation_type
        detected_types[vt] = detected_types.get(vt, 0) + 1

    print(f"\nDetected per type: {detected_types}\n")

    # Compute per-type metrics
    hdr = f"{'Violation Type':<40} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}"
    print(hdr)
    print("-" * len(hdr))

    all_p, all_r, all_f1 = [], [], []
    per_type_out = {}

    for vtype, gt_count in GROUND_TRUTH.items():
        det_count = detected_types.get(vtype, 0)
        tp = min(det_count, gt_count)
        fp = max(0, det_count - gt_count)
        fn = max(0, gt_count - det_count)
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt_count == 0 else 0.0)
        recall    = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if gt_count == 0 else 0.0)
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if gt_count > 0:
            all_p.append(precision)
            all_r.append(recall)
            all_f1.append(f1)

        per_type_out[vtype] = {
            "ground_truth": gt_count, "detected": det_count,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        }
        print(f"{vtype:<40} {gt_count:>4} {det_count:>4} {tp:>4} {fp:>4} {fn:>4} "
              f"{precision:>6.1%} {recall:>6.1%} {f1:>6.1%}")

    macro_p  = sum(all_p)  / len(all_p)  if all_p  else 0.0
    macro_r  = sum(all_r)  / len(all_r)  if all_r  else 0.0
    macro_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0

    print("-" * len(hdr))
    print(f"{'MACRO AVERAGE':<40} {'':>4} {'':>4} {'':>4} {'':>4} {'':>4} "
          f"{macro_p:>6.1%} {macro_r:>6.1%} {macro_f1:>6.1%}")

    print(f"""
NOTE: Evaluated on offline synthetic dataset only.
Production metrics require real BTP CCTV footage + manual annotations.
Macro P={macro_p:.1%}  R={macro_r:.1%}  F1={macro_f1:.1%}
""")

    metrics = {
        "detector": detector,
        "dataset": "synthetic_stage1.mp4 (150 frames @ 18fps)",
        "frames_processed": result["frames_processed"],
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_type": per_type_out,
        "note": (
            "Evaluated on offline synthetic dataset (create_synthetic_video.py). "
            "Production evaluation requires real BTP CCTV footage with human-annotated ground truth."
        ),
    }

    out = ROOT / "sample_data" / "eval_output" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Full metrics saved -> {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChalanReady AI offline evaluation")
    parser.add_argument("--detector", default="color",
                        choices=["auto", "color", "yolo"])
    args = parser.parse_args()
    evaluate(args.detector)
