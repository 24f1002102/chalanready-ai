# ChalanReady AI
## Automated Traffic Violation Detection & Officer Review System
### Flipkart Gridlock Hackathon 2.0 - Problem Statement 3
**Bengaluru Traffic Police - ASTraM Unit - June 2026**

---

## 1. Executive Summary

ChalanReady AI is an **officer-in-the-loop** AI pipeline that automatically detects, classifies, and documents traffic violations from CCTV footage - while ensuring that **every enforcement action requires explicit human officer approval**. No challan is ever issued automatically.

> *"ChalanReady AI is not an autonomous enforcement robot. It is a force multiplier for traffic officers - turning 1 officer's reach into the reach of 50 cameras, while keeping every enforcement decision in human hands."*

---

## 2. Problem Being Solved

Manual inspection of traffic camera footage is:
- **Labour-intensive**: Bengaluru has 5,000+ CCTV cameras; monitoring all is impossible with current manpower
- **Inconsistent**: Human fatigue and attention drift cause missed violations
- **Slow**: Violations are rarely caught in real-time; most evidence is found retrospectively
- **Evidence-weak**: Without timestamped annotated frames, violations are hard to prosecute

ChalanReady AI solves all four problems simultaneously.

---

## 3. System Architecture

```text
CCTV Camera / Uploaded Video
  -> Image preprocessing (CLAHE + Gaussian denoise)
  -> Vehicle detection (YOLOv8n local, color-HSV offline fallback)
  -> Centroid multi-object tracking
  -> Camera-calibrated violation rule engine
     - wrong-side driving: trajectory direction vs configured lane vector
     - illegal parking: no-parking polygon + anchored dwell time
     - footpath riding: footpath polygon entry
     - red-light/stop-line: calibrated signal ROI and stop line
     - helmet/seatbelt/triple-riding: experimental review hints, disabled by default
  -> Plate reading (EasyOCR when available, labelled fallback otherwise)
  -> Evidence packet (annotated frame, timestamp, camera profile, confidence)
  -> Officer review queue (approve, reject, flag)
  -> Analytics and reporting
```

---

## 4. Violation Types Detected

| Violation | Detection Method | Prototype Status |
|-----------|------------------|------------------|
| Wrong-side driving | Camera-calibrated trajectory direction | Active on calibrated profiles |
| Illegal parking | Restricted polygon + anchored 60-frame dwell | Active on calibrated profiles |
| Footpath riding | Footpath polygon entry count | Active on calibrated profiles |
| Red-light / stop-line violation | Calibrated stop line + signal ROI | Active on calibrated profiles |
| Helmet / seatbelt / triple riding | Heuristic review hints | Experimental, disabled by default |

---

## 5. Key Innovation: The Officer-in-the-Loop Model

Most automated traffic systems attempt full automation. ChalanReady AI deliberately does not.

**Why this matters:**
- **Legal**: In India, traffic challans require an authorizing officer. Automated fines without human sign-off face legal challenges.
- **Accuracy**: Even a 95% accurate AI has a 5% false positive rate. At 1,000 violations/day, that's 50 wrongful challans. Officers catch these.
- **Trust**: Citizens and courts accept officer-reviewed evidence far more readily than purely automated evidence.
- **Operational fit**: BTP officers currently review footage retrospectively. ChalanReady AI makes their review proactive and systematic - same workflow, 10x efficiency.

**The force multiplier effect:**
- Without AI: 1 officer can monitor ~3 cameras in real-time
- With ChalanReady AI: 1 officer reviews a pre-filtered queue covering 50+ cameras, acting only on flagged candidates

---

## 6. Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Detection | YOLOv8n (Ultralytics) | State-of-art, real-time, lightweight |
| Tracking | Custom Centroid Tracker | Zero dependencies, privacy-safe |
| Backend API | FastAPI (Python) | High-performance async, auto OpenAPI docs |
| Data models | Pydantic v2 | Type-safe, schema-validated evidence packets |
| Frontend | Vanilla HTML/CSS/JS + Chart.js | No build step, deployable anywhere |
| Image processing | OpenCV | Industry standard for video pipelines |
| OCR | EasyOCR (planned) | Best open-source Indian plate recognition |
| Map Integration | MapMyIndia SDK | Official partner - India's mapping infrastructure |

---

## 7. MapMyIndia Integration (Partner)

ChalanReady AI is designed to integrate with the official hackathon partner **MapMyIndia**:

- **Violation GPS pinning**: Each camera has a configured lat/lng; violations are pinned on a MapMyIndia live map
- **Zone boundary overlay**: No-parking zones and footpath boundaries drawn using MapMyIndia polygon APIs
- **Patrol route optimization**: MapMyIndia routing API used to suggest officer patrol routes based on violation hotspots
- **Heatmap tiles**: Aggregated violation density rendered as a MapMyIndia heatmap layer

The `ViolationPacket` schema already includes `gps_lat` and `gps_lng` fields ready for this integration.

---

## 8. Scalability Design

| Concern | Current (Prototype) | Production Plan |
|---------|--------------------|-|
| Storage | In-memory dict | PostgreSQL / TimescaleDB |
| Processing | Synchronous per-video | Celery + Redis task queue |
| Cameras | Single file upload | RTSP stream ingestion |
| Officers | Single user | Role-based auth (BTP officer IDs) |
| Scale | Local | Docker + Kubernetes on cloud |

The FastAPI + Pydantic + async design makes the transition to production straightforward.

---

## 9. Synthetic Smoke-Test Metrics

Evaluated with `sample_data/evaluate.py` on an offline scripted synthetic CCTV clip:

| Metric | Score | Scope |
|--------|-------|-------|
| Precision | 100% | Synthetic smoke test: 4 scripted violation events |
| Recall | 100% | Synthetic smoke test: 4 scripted violation events |
| F1-Score | 100% | Synthetic smoke test: 4 scripted violation events |
| Processing speed | ~18 FPS | Synthetic CPU demo |

These numbers validate pipeline behavior on known scripted events, not production field accuracy. Production metrics still require real BTP CCTV footage with human-annotated ground truth.

---

## 10. Demo

**Running the prototype:**
```bash
# Start the system (one command)
start.bat

# Dashboard opens at: http://127.0.0.1:8000
# API docs at:        http://127.0.0.1:8000/docs
```

**Dashboard features:**
1. **Dashboard**: Real-time stats, violation charts, activity feed
2. **Review Queue**: All pending violations with Approve / Reject / Flag actions
3. **Live Feed**: Annotated violation snapshot thumbnails
4. **Analytics**: Confidence distribution, hourly timeline, zone heatmap
5. **Violation Map**: Geographic hotspot canvas (MapMyIndia integration-ready)
6. **AI Metrics**: Precision / Recall / F1 / mAP per violation type
7. **Upload Video**: Drag-and-drop video processing with live log

---

## 11. What Makes This Submission Stand Out

1. **Ethical AI by design** - officer approval gate is not a feature; it's the architecture
2. **4 calibrated validation events implemented** - wrong-side, parking, footpath, red-light/stop-line
3. **Full-stack working prototype** - backend API + frontend dashboard + video pipeline
4. **Evidence-grade outputs** - timestamped annotated JPEGs, Indian plate numbers, zone metadata
5. **Partner integration ready** - MapMyIndia GPS fields and schema designed for their APIs
6. **Real-world deployment plan** - Docker, PostgreSQL, RTSP streams, role-based auth mapped out

---

*Submitted for Flipkart Gridlock Hackathon 2.0 - Problem Statement 3*
*Bengaluru Traffic Police - ASTraM Unit - June 2026*
