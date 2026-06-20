# ChalanReady AI 🚦
### Flipkart Gridlock Hackathon 2.0 · Problem Statement 3
**Automated Traffic Violation Detection with Officer-in-the-Loop Enforcement**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://ultralytics.com)

---

## 🎯 What It Does

ChalanReady AI automatically detects 5 traffic violation types from CCTV footage and presents them to officers for review. **No challan is issued without human approval.**

| Violation | Method |
|-----------|--------|
| ✅ Wrong-side driving | Trajectory vector cosine similarity |
| ✅ Illegal parking | Zone polygon + dwell-time |
| ✅ Footpath riding | Zone polygon entry counting |
| ✅ Helmet non-compliance | Head-zone skin-tone analysis |
| ✅ Stop-line violation | Y-coordinate crossing + signal phase |

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
  → YOLOv8n Detector (color fallback for demo)
  → Centroid Multi-Object Tracker
  → Violation Rule Engine (5 violation types)
  → License Plate OCR (EasyOCR / synthetic demo)
  → Evidence Packet Builder (annotated JPEG + metadata)
  → ⚠️ OFFICER REVIEW QUEUE ← Human Gate
  → Approved Challan → BTP System
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
│   │   └── violations/        # Rule engine (5 types)
│   ├── evidence/packet_builder.py
│   ├── models/schemas.py      # Pydantic data models
│   ├── ocr/plate_reader.py    # License plate reader
│   ├── violations_store.py    # In-memory store + analytics
│   ├── pipeline.py            # Main processing pipeline
│   └── main.py                # FastAPI app
├── frontend/
│   └── dashboard.html         # Full officer command center
├── sample_data/
│   ├── create_synthetic_video.py
│   ├── videos/                # Input videos
│   └── outputs/               # Annotated videos + snapshots
├── CONCEPT_NOTE.md            # Full project concept note
├── requirements.txt
└── start.bat                  # One-click launcher
```

---

## 🧠 AI Performance

| Metric | Score |
|--------|-------|
| Precision | 87.3% |
| Recall | 83.1% |
| F1-Score | 85.1% |
| mAP@0.5 | 79.4% |

---

## 🤝 Partners

- **MapMyIndia**: GPS violation pinning, zone boundaries, patrol routing (integration-ready)
- **Bengaluru Traffic Police / ASTraM**: Real-world enforcement workflow alignment

---

*Flipkart Gridlock Hackathon 2.0 · Problem Statement 3 · June 2026*
