from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .api.routes import router

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="ChalanReady AI",
    description=(
        "Officer-in-the-loop AI pipeline for traffic violation detection. "
        "Detects wrong-side driving, illegal parking, footpath riding, "
        "stop-line violations, and helmet non-compliance. "
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
