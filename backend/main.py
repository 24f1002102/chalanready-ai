from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import threading

from .api.routes import router

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Real CCTV demo video uploaded by the user.
# Falls back to the synthetic video if the real one is not present.
_REAL_VIDEO = PROJECT_ROOT / "sample_data" / "videos" / \
    "stock-footage-delhi-india-jul-smooth-traffic-flow-at-intersection-with-green-signal.mov"
_SYNTHETIC_VIDEO = PROJECT_ROOT / "sample_data" / "videos" / "synthetic_stage1.mp4"


def _get_demo_video() -> Path | None:
    if _REAL_VIDEO.exists():
        return _REAL_VIDEO
    if _SYNTHETIC_VIDEO.exists():
        return _SYNTHETIC_VIDEO
    return None


def _auto_process_demo() -> None:
    """
    Optional local helper: run the demo pipeline at startup.
    Disabled by default so judge/user uploads are never mixed with implicit seed data.
    """
    if os.getenv("CHALANREADY_STARTUP_DEMO") != "1":
        return

    try:
        from .violations_store import store
        analytics = store.get_analytics()
        if analytics["total_violations"] > 0:
            return  # already have real data, skip

        demo_video = _get_demo_video()
        if demo_video is None:
            return  # no video to process

        from .pipeline import process_video
        output_dir = PROJECT_ROOT / "sample_data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"demo_annotated_{demo_video.stem}.mp4"

        # Use real CCTV zone for the real video; synthetic zone for fallback
        zone = "Zone-A / MG Road"
        detector = "auto"  # uses YOLO if available, color-HSV as fallback

        print(f"[ChalanReady AI] Auto-processing demo video: {demo_video.name}")
        result = process_video(
            input_path=demo_video,
            output_path=output_path,
            detector_backend=detector,
            zone_name=zone,
            store=store,
        )
        print(
            f"[ChalanReady AI] Demo pipeline complete. "
            f"Frames: {result['frames_processed']}, "
            f"Violations: {result['violations_detected']}, "
            f"Detector: {result['detector']}"
        )
    except Exception as exc:
        print(f"[ChalanReady AI] Auto-process warning: {exc}")


app = FastAPI(
    title="ChalanReady AI",
    description=(
        "Officer-in-the-loop AI pipeline for traffic violation detection. "
        "Detects wrong-side driving, illegal parking, footpath riding, "
        "stop-line/red-light violations, helmet non-compliance, seatbelt "
        "non-compliance, and triple riding. "
        "Every enforcement action requires human officer approval."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve violation snapshot images
SNAPSHOTS_DIR = PROJECT_ROOT / "sample_data" / "outputs" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

# Serve the officer dashboard UI at /ui/
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the officer dashboard."""
    return RedirectResponse(url="/ui/dashboard.html")


@app.on_event("startup")
async def startup_event():
    """Start with a clean runtime path; demo processing is explicit from the UI."""
    if os.getenv("CHALANREADY_STARTUP_DEMO") == "1":
        thread = threading.Thread(target=_auto_process_demo, daemon=True)
        thread.start()
