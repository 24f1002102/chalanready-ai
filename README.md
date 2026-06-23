# ChalanReady AI
### Flipkart Gridlock Hackathon 2.0 · Problem Statement 3
**AI-Powered Traffic Violation Detection with Officer-in-the-Loop Enforcement**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://ultralytics.com)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7+-orange)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Live Demo

🌐 **Deployed on Railway** → https://web-production-477cc.up.railway.app/ui/dashboard.html

Note: YOLO is intentionally left out of that file: If you look at your requirements.txt, you'll see this note at the top:
text
*Cloud deployment requirements (no ultralytics = no opencv-GUI conflict)*
*The app auto-falls back to the color detector in cloud/demo mode.*
The ultralytics package (which powers YOLOv8) is massive. It requires a lot of RAM and storage, which often causes free-tier cloud deployments(like Railway) to crash or timeout. However you can run the app locally in your desktop.

Open dashboard -> click **Run Demo Pipeline** -> explore the full pipeline.

---

## What It Does

ChalanReady AI continuously monitors CCTV footage, detects traffic violations using computer vision, and presents them to traffic officers for review. **No challan is issued without explicit human approval** (Officer-in-the-Loop principle).

### Supported Violation Types

| Violation | Detection Method | Status |
|-----------|-----------------|--------|
| Wrong-Side Driving | Trajectory vector cosine similarity | ✅ Active |
| Illegal / Footpath Parking | Zone polygon + dwell-time gate (60 frames) | ✅ Active |
| Footpath Riding | Zone polygon entry counting | ✅ Active |
| Red-Light Violation | Stop-line crossing during red phase | ✅ Active |
| Stop-Line Violation | Stop-line geometry crossing | ✅ Active |
| Helmet Non-Compliance | Head-zone HSV heuristic | 🔬 Experimental |
| Seatbelt Non-Compliance | Pixel / pose heuristic | 🔬 Experimental |
| Triple Riding | Multi-person crop analysis | 🔬 Experimental |

---

## Quick Start

```bat
start.bat
```
Opens the dashboard at **http://127.0.0.1:8000** automatically.

---

## Manual Setup

```powershell
cd chalanready-ai
python -m venv .venv
.venv\Scripts\Activate.ps1

# Base install (cloud/demo mode)
pip install -r requirements.txt

# Full install (includes YOLOv8 + EasyOCR for real detection)
pip install -r requirements-local.txt
```

### Generate synthetic demo video
```powershell
python sample_data/create_synthetic_video.py
```

### Run offline evaluation (Precision / Recall / F1)
```powershell
python sample_data/evaluate.py
# or with YOLOv8:
python sample_data/evaluate.py --detector yolo
```

### Start API + Dashboard
```powershell
uvicorn backend.main:app --reload
# Dashboard: http://127.0.0.1:8000
# API docs:  http://127.0.0.1:8000/docs
```

### Build submission ZIP
```powershell
python pack.py
# Creates: chalanready_submit.zip
```

---

## Dashboard Features

| Screen | Description |
|--------|-------------|
| **Command Center** | Live stats, violation charts, ₹ fine revenue estimate, activity feed |
| **Review Queue** | Approve / Reject / Flag with one click · Click any row to view evidence snapshot |
| **Live Feed** | Annotated violation snapshot thumbnails |
| **Analytics** | Confidence distribution, hourly timeline, zone heatmap |
| **Violation Map** | Leaflet map with dynamic zone pins from camera profiles API |
| **AI Metrics** | Precision / Recall / F1 per violation type (from `evaluate.py`) |
| **Upload Video** | Drag-and-drop video processing with real-time log |
| **Image Analyze** | Single-frame violation detection |

---

## Architecture

```
Video File / RTSP Stream
  → Stage 1: CLAHE Preprocessing + Gaussian Denoise
  → YOLOv8n Detector (color-HSV fallback for synthetic demo)
  → Centroid Multi-Object Tracker
  → Violation Rule Engine (camera-calibrated geometry rules)
  → License Plate OCR (EasyOCR if installed · honest "PLATE_UNREAD" fallback)
  → Evidence Packet Builder (annotated JPEG snapshot + full metadata)
  → OFFICER REVIEW QUEUE  ← Human Gate (no auto-challan ever)
  → Approved Challan → CSV Export → BTP System
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | System health check |
| `GET`  | `/api/cameras` | List calibrated camera profiles (dynamic) |
| `POST` | `/api/images/analyze` | Upload & analyze a single traffic photo |
| `POST` | `/api/videos/process` | Upload & process video file |
| `GET`  | `/api/violations` | List violations (filterable by status/type) |
| `POST` | `/api/violations/{id}/review` | Officer approve / reject / flag |
| `GET`  | `/api/analytics` | Full analytics payload for dashboard |
| `GET`  | `/api/analytics/summary` | Quick summary (peak hour, hottest zone, revenue) |
| `GET`  | `/api/analytics/export` | Download violations as CSV |
| `GET`  | `/api/ai/status` | YOLO/OCR readiness and mode transparency |
| `GET`  | `/api/metrics` | Synthetic smoke-test metrics (Precision / Recall / F1) |
| `POST` | `/api/demo/seed` | Run demo pipeline without clearing existing evidence |
| `GET`  | `/snapshots/{filename}` | Serve evidence snapshot image |

---

## Camera Profiles

12 camera profiles included (`backend/config/camera_profiles.json`), including synthetic/demo profiles, one generic upload profile, and calibrated Bengaluru zones:

| Camera ID | Location | Enabled Rules |
|-----------|----------|---------------|
| `synthetic_stage1` | Synthetic Validation | All |
| `delhi_intersection_demo` | Delhi Demo Clip | wrong_side, stopline, helmet |
| `bengaluru_generic_upload` | Generic Upload | All |
| `zone_b_indiranagar` | Indiranagar 100ft Rd | All |
| `zone_c_koramangala` | Koramangala 5th Block | wrong_side, parking, footpath, helmet, triple |
| `zone_d_whitefield` | Whitefield Main Rd | wrong_side, parking, seatbelt, stopline |
| `zone_e_jayanagar` | Jayanagar 4th Block | All |
| `zone_f_electronic_city` | Electronic City Ph1 | wrong_side, parking, seatbelt, stopline, helmet |
| `zone_g_silk_board` | Silk Board Junction | wrong_side, parking, footpath, helmet, stopline |
| `zone_h_hebbal` | Hebbal Flyover | wrong_side, parking, seatbelt, stopline |
| `zone_i_majestic` | Majestic Bus Stand | wrong_side, parking, footpath, helmet, triple, stopline |
| `zone_j_marathahalli` | Marathahalli Bridge | wrong_side, parking, seatbelt, helmet, stopline |

---

## Synthetic Smoke-Test Metrics

Evaluated on an **offline scripted synthetic validation clip** (`python sample_data/evaluate.py`):

> ⚠ These metrics are on a controlled synthetic dataset only.
> These numbers validate pipeline behavior, not production field accuracy.
> Production evaluation requires real BTP CCTV footage with human-annotated ground truth.
> Run `python sample_data/evaluate.py` to reproduce exact numbers.

| Metric | Score | Dataset |
|--------|-------|---------|
| Macro Precision | Computed at runtime | Synthetic — 4 scripted events |
| Macro Recall | Computed at runtime | Synthetic — 4 scripted events |
| Macro F1 | Computed at runtime | Synthetic — 4 scripted events |

Cached metrics are saved to `sample_data/eval_output/metrics.json` after first run.
Live metrics available at `/api/metrics`.

---

## License Plate OCR

- **With EasyOCR** (`pip install easyocr`): Real plate recognition on vehicle bbox crops
- **Without EasyOCR**: Returns `PLATE_UNREAD` with `confidence=0.0` and `source=ocr_unavailable`
- The UI shows **"⚠ OCR N/A"** — never a fake plate number
- Set env `CHALANREADY_OCR_MODE=real` to force EasyOCR mode

---

## Project Structure

```
chalanready-ai/
 backend/
    api/routes.py           All API endpoints
    config/
       camera_profiles.json  12 camera profiles (GPS + geometry)
       cameras.py            Profile loader + CameraProfile dataclass
    detection/
       detector.py           YOLOv8 + color-HSV detector
       tracker.py            Centroid multi-object tracker
       violations/           Rule engine (6 independent rules)
    evidence/packet_builder.py
    models/schemas.py        Pydantic models
    ocr/plate_reader.py      EasyOCR + honest fallback
    violations_store.py      SQLite store + analytics + CSV export
    pipeline.py              Main processing pipeline
    main.py                  FastAPI app
 frontend/
    dashboard.html           Officer command center (single-file SPA)
 sample_data/
    create_synthetic_video.py  Scripted test video generator
    evaluate.py                Reproducible P/R/F1 evaluation
    videos/                    Input videos
 pack.py                     Submission ZIP builder
 requirements.txt            Cloud deployment
 requirements-local.txt      Local dev (YOLOv8 + EasyOCR)
 start.bat                   One-click launcher
```

---

## Partners

- **MapMyIndia**: GPS violation pinning, zone boundaries, patrol routing (integration-ready)
- **Bengaluru Traffic Police / ASTraM**: Real-world enforcement workflow alignment

---

*Flipkart Gridlock Hackathon 2.0 · Problem Statement 3 · June 2026*
