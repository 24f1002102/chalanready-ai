# Sample Data

Stage 1 includes an offline synthetic clip generator so the pipeline can be tested without internet access or a live traffic camera.

Generate it with:

```powershell
python sample_data/create_synthetic_video.py
```

This creates:

```text
sample_data/videos/synthetic_stage1.mp4
```

For a stronger demo, add one or two real fixed-camera clips under `sample_data/videos/` and run the pipeline with a local YOLOv8 weight file:

```powershell
python -m backend.pipeline --input sample_data/videos/real_clip.mp4 --detector yolo
```

Useful clip traits:

- Static CCTV-like camera
- Visible lanes or road direction
- At least one stopped vehicle in a no-parking/footpath-like region
- At least one two-wheeler or car crossing into a pedestrian edge area
- Short clips, ideally 10 to 30 seconds

Known demo risk: wrong-side detection depends on a configured lane direction. A clip with a curved road, moving camera, or ambiguous lane flow will need manual zone calibration before the rule is credible.
