# ChalanReady AI 🚦
### Flipkart Gridlock Hackathon 2.0 · Problem Statement 3
**Automated Traffic Violation Detection with Officer-in-the-Loop Enforcement**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://ultralytics.com)

---

## 🎯 What It Does

ChalanReady AI automatically detects traffic violations from CCTV footage and presents them to officers for review. **No challan is issued without human approval.**

**3 fully-implemented violation types** (rule-engine verified on synthetic dataset):

| Violation | Method | Status |
|-----------|--------|--------|
| Wrong-side driving | Trajectory vector cosine similarity | **Live** |
| Illegal parking | Zone polygon + dwell-time (45 frames) | **Live** |
| Footpath riding | Zone polygon entry counting | **Live** |
| Helmet non-compliance | Head-zone HSV skin-tone heuristic | Live (heuristic) |
| Stop-line violation | Y-coordinate crossing + signal phase | Requires red-phase input |
| Seatbelt non-compliance | Pose-based keypoint analysis | Stub — needs YOLOv8-pose |
| Triple riding | Multi-person head detection | Stub — needs labeled dataset |

> **Evaluation**: Run `python sample_data/evaluate.py` for reproducible Precision / Recall / F1 on the synthetic dataset.

---

## 🚀 Quick Start (One Command)

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
pip install -r requirements.txt
```

### Generate demo video
```powershell
python sample_data/create_synthetic_video.py
```

### Run AI pipeline
```powershell
python -m backend.pipeline --input sample_data/videos/synthetic_stage1.mp4 --detector color
```

### Start API + Dashboard
```powershell
uvicorn backend.main:app --reload
# Dashboard: http://127.0.0.1:8000
# API docs:  http://127.0.0.1:8000/docs
```

---

## 📊 Dashboard Features

| Screen | Description |
|--------|-------------|
| **Dashboard** | Live stats, violation charts, activity feed |
| **Review Queue** | Approve / Reject / Flag violations with one click |
| **Live Feed** | Annotated violation snapshot thumbnails |
| **Analytics** | Confidence distribution, hourly timeline, zone heatmap |
| **Violation Map** | Geographic hotspot canvas (MapMyIndia integration ready) |
| **AI Metrics** | Precision / Recall / F1 / mAP per violation type |
| **Upload Video** | Drag-and-drop processing with real-time log |

---

## 🏗️ Architecture

```
Video File / RTSP Stream
  -> Stage 1: CLAHE Preprocessing + Gaussian Denoise
  -> YOLOv8n Detector (color-HSV fallback for synthetic demo)
  -> Centroid Multi-Object Tracker
  -> Violation Rule Engine (3 active types + 2 documented stubs)
  -> License Plate OCR (EasyOCR if installed / synthetic demo fallback)
  -> Evidence Packet Builder (annotated JPEG + metadata)
  -> OFFICER REVIEW QUEUE  <-- Human Gate (no auto-challan)
  -> Approved Challan -> BTP System
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | System health |
| `POST` | `/api/videos/process` | Upload & process video |
| `GET`  | `/api/violations` | List violations (filterable) |
| `POST` | `/api/violations/{id}/review` | Officer approve/reject/flag |
| `GET`  | `/api/analytics` | Full analytics payload |
| `POST` | `/api/demo/seed` | Seed demo violations |
| `GET`  | `/snapshots/{filename}` | Serve evidence image |

---

## 📁 Project Structure

```
chalanready-ai/
├── backend/
│   ├── api/routes.py          # All API endpoints
│   ├── detection/
│   │   ├── detector.py        # YOLOv8 + color detector
│   │   ├── tracker.py         # Centroid tracker
│   │   └── violations/        # Rule engine (3 active + 2 stubs)
│   ├── evidence/packet_builder.py
│   ├── models/schemas.py      # Pydantic data models
│   ├── ocr/plate_reader.py    # EasyOCR + synthetic demo fallback
│   ├── violations_store.py    # In-memory store + analytics
│   ├── pipeline.py            # Main processing pipeline + CLAHE
│   └── main.py                # FastAPI app
├── frontend/
│   └── dashboard.html         # Full officer command center
├── sample_data/
│   ├── create_synthetic_video.py
│   ├── evaluate.py            # Reproducible eval script
│   ├── videos/                # Input videos
│   └── outputs/               # Annotated videos + snapshots
├── CONCEPT_NOTE.md            # Full project concept note
├── requirements.txt           # Cloud deployment (no ultralytics)
├── requirements-local.txt     # Local dev (adds ultralytics/YOLOv8)
└── start.bat                  # One-click launcher
```

---

## 🧠 AI Performance

Evaluated on offline synthetic dataset (`sample_data/evaluate.py`):

| Metric | Score | Dataset |
|--------|-------|---------|
| Precision (macro) | 100% | Synthetic (3 violations, controlled GT) |
| Recall (macro) | 100% | Synthetic (3 violations, controlled GT) |
| F1-Score (macro) | 100% | Synthetic (3 violations, controlled GT) |

> **Note**: Synthetic dataset metrics reflect controlled conditions.
> Production evaluation requires real BTP CCTV footage with human-annotated ground truth.
> Run `python sample_data/evaluate.py` to reproduce.

---

## 🤝 Partners

- **MapMyIndia**: GPS violation pinning, zone boundaries, patrol routing (integration-ready)
- **Bengaluru Traffic Police / ASTraM**: Real-world enforcement workflow alignment

---

*Flipkart Gridlock Hackathon 2.0 · Problem Statement 3 · June 2026*
