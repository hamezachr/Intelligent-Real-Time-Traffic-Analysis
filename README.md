# Intelligent Real‑Time Traffic Analysis (YOLO + OpenCV)

An end‑to‑end, real‑time traffic analytics system powered by YOLO (Ultralytics) and OpenCV. It detects and tracks people and vehicles, shows live counts per class, and saves detailed track data for later analysis. The pipeline supports camera streams and video files, runs headless (no GUI) when needed, and can leverage CUDA GPUs for speed.

## Features
- Real‑time detection of `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `train` (COCO classes)
- Persistent multi‑object tracking with unique IDs
- Live overlay: class labels and counts per frame
- CSV logging per frame per track for downstream analytics
- Camera and video input, headless mode, optional output recording
- CUDA acceleration when PyTorch with CUDA is installed

## Requirements
- Python `3.9+`
- Dependencies: `pip install -r requirements.txt`
- Optional GPU: install a PyTorch CUDA build for your GPU to use `--device cuda`

## Quick Start
- Camera (index `0`):
  - `python traffic_analyzer.py --source 0`
- Video file:
  - `python traffic_analyzer.py --source .\traffic.mp4`
- Headless and record output:
  - `python traffic_analyzer.py --source 0 --no_show --out_video output.mp4`
- Choose classes:
  - `python traffic_analyzer.py --source .\traffic.mp4 --classes person,bicycle,car,motorcycle,bus,truck,train`
- Use CUDA if available:
  - `python traffic_analyzer.py --source .\traffic.mp4 --device cuda`

## Configuration
- `--source` input source (`0` for default camera or a video path)
- `--model` YOLO model file (`yolov8n.pt` by default)
- `--device` `auto|cuda|cpu` (auto uses CUDA if available)
- `--no_show` run without GUI windows (recommended on servers or if GUI fails)
- `--out_video` path to save the processed video (e.g., `output.mp4`)
- `--max_frames` limit processing to a fixed number of frames (useful for tests)
- `--classes` comma‑separated list to restrict detection (defaults to a standard set)
- `--conf` detection confidence threshold (default `0.25`)
- `--iou` NMS IoU threshold (default `0.45`)
- `--meters_per_pixel` scene calibration for speed estimation in logs (default `0.05`)

## Output and Logging
- On‑screen overlay (when GUI is available): per‑object class label and ID, top‑left counters per class
- Headless mode: same processing, no windows; optional saved video via `--out_video`
- CSV file: `traffic_log.csv` stores track records with:
  - `time, frame, track_id, class, class_fr, speed_mps, speed_kmh, x1, y1, x2, y2`
  - Speed values depend on `--meters_per_pixel` and video `fps`. Overlays no longer show km/h, but the CSV still logs speeds for analysis.

## Calibration (Optional)
- Speed estimation relies on `meters_per_pixel`. Measure a known real‑world distance visible in the frame (e.g., lane width in meters) and the corresponding pixel distance, then set:
  - `meters_per_pixel = real_distance_m / pixel_distance`

## CUDA Acceleration
- Install a PyTorch build matching your CUDA version, then run with `--device cuda`.
- Example (adjust to your CUDA version):
  - `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- When `--device auto` is used, the app selects CUDA if available, otherwise CPU. Half‑precision inference is enabled on CUDA for speed.

## Troubleshooting
- OpenCV GUI errors (e.g., `cv2.imshow`/`cv2.waitKey` not implemented):
  - Use headless mode: `--no_show`, and optionally `--out_video output.mp4`
- GPU not used:
  - Ensure a CUDA‑enabled PyTorch wheel is installed; verify with a simple script:
    - `import torch; print(torch.cuda.is_available())`
- Video does not open:
  - Check the path/format, confirm codec support; try a different file or camera index

## Project Structure
- `traffic_analyzer.py` main application (input, detection, tracking, overlays, logging)
- `src/yolo_detector.py` YOLO detection wrapper with device/half settings
- `src/tracker.py` simple IoU tracker with ID maintenance and EMA speed smoothing
- `requirements.txt` project dependencies


