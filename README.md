# IndustrialEye AI

AI-powered production detection, tracking, and counting for manufacturing or conveyor-belt footage.

This is a functional Streamlit POC. A user uploads production-line video, the app detects objects with Ultralytics YOLO, tracks each physical object with ByteTrack, counts it once when it crosses a virtual line or counting zone, writes an annotated output video, and exports analytics.

## What The POC Demonstrates

A manufacturer uploads CCTV or conveyor footage. The AI analyzes each frame, assigns persistent tracking IDs, and counts a package/product only after its tracked center point crosses the configured counting line or ROI zone. Management receives total production, direction counts, throughput, event logs, confidence charts, CSV exports, and processed video evidence.

## Architecture

```text
industrialeye-ai/
app.py
requirements.txt
README.md
.gitignore
assets/
  styles.css
models/
  .gitkeep
temp/
  .gitkeep
outputs/
  .gitkeep
samples/
  sample_conveyor.mp4
src/
  config.py
  detection/
    model_loader.py
    detector.py
  tracking/
    tracker.py
  counting/
    line_counter.py
  processing/
    video_processor.py
  analytics/
    metrics.py
  ui/
    components.py
    styles.py
  utils/
    video_utils.py
    file_utils.py
```

## Installation

Python 3.11+ is recommended.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Push To GitHub

Runtime files are excluded with `.gitignore`, including model weights, uploaded videos, processed videos, Python caches, and local environment files.

Create the first local commit:

```bash
git init
git add .
git commit -m "Initial IndustrialEye AI POC"
```

Create an empty repository on GitHub, then connect this folder to it:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/industrialeye-ai.git
git push -u origin main
```

For later changes:

```bash
git add .
git commit -m "Update IndustrialEye AI"
git push
```

Do not commit downloaded model files such as `models/yolo11n.pt`, custom models such as `models/manufacturing_best.pt`, uploaded videos, or processed output videos.

The repository includes one small demo video at:

```text
samples/sample_conveyor.mp4
```

That bundled sample allows the deployed app to start analysis immediately without requiring an upload first.

## Server Deployment

These commands assume a Linux server with Python 3.11+, git, and ffmpeg available.

Install system packages:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git ffmpeg
```

Clone the project:

```bash
git clone https://github.com/YOUR_USERNAME/industrialeye-ai.git
cd industrialeye-ai
```

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run the app on the server:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open this in the browser:

```text
http://SERVER_IP:8501
```

For a background process without systemd:

```bash
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
```

To stop that background process:

```bash
pkill -f "streamlit run app.py"
```

### Optional systemd Service

Create a service file:

```bash
sudo nano /etc/systemd/system/industrialeye-ai.service
```

Paste this, replacing `ubuntu` and `/home/ubuntu/industrialeye-ai` with your server user/path:

```ini
[Unit]
Description=IndustrialEye AI Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/industrialeye-ai
Environment="PATH=/home/ubuntu/industrialeye-ai/venv/bin"
ExecStart=/home/ubuntu/industrialeye-ai/venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable industrialeye-ai
sudo systemctl start industrialeye-ai
sudo systemctl status industrialeye-ai
```

View logs:

```bash
journalctl -u industrialeye-ai -f
```

## Models

The default model is `yolo11n.pt`, a lightweight Ultralytics pretrained YOLO model suitable for CPU POC execution. The app loads it through Ultralytics and downloads official weights on first use under:

```text
models/
```

Large model files are ignored by git.

## Detection Modes

### Standard YOLO

Uses the lightweight pretrained YOLO model for common COCO-style classes. This is useful for objects like bottle, person, box-like classes covered by the pretrained model vocabulary, and general experimentation.

### Open Vocabulary

Uses `yolov8s-worldv2.pt` and attempts to set target prompts such as:

```text
box, package, carton, bottle, product, component
```

If the installed Ultralytics version or model cannot apply open-vocabulary prompts, the app shows a warning and falls back to the model default vocabulary. It never fabricates detections.

### Custom Manufacturing Model

Place a trained model at:

```text
models/manufacturing_best.pt
```

or upload a `.pt` file from the sidebar. The same detection, tracking, counting, annotation, and analytics pipeline will be used without rewriting the processing code.

## Detection vs Tracking vs Counting

Detection finds objects in a frame and returns bounding boxes, classes, and confidence scores.

Tracking links detections across frames and assigns a persistent `track_id` to each physical object.

Counting uses the tracked center point history. It compares the previous center to the current center and counts an object only when the track crosses the configured virtual line or enters the ROI zone.

Do not add the number of detections on every frame. The same physical carton can appear in hundreds of frames, so per-frame detection counting would massively overcount. This app maintains:

```python
counted_track_ids = set()
```

After a track crosses the counting line:

```python
if track_id not in counted_track_ids:
    total_count += 1
    counted_track_ids.add(track_id)
```

## Counting Modes

Horizontal Line counts vertical movement across a horizontal line.

Vertical Line counts horizontal movement across a vertical line.

ROI Zone counts when a tracked center enters a horizontal counting band.

Direction can be `Both`, `Forward`, or `Reverse`. The app maintains forward, reverse, and total counts separately.

## Outputs

After analysis the app provides:

- Browser-previewable processed video
- Detection event CSV
- Summary CSV
- Objects counted over time chart
- Production rate over time chart
- Object distribution chart
- Confidence distribution chart

Event CSV columns:

```text
timestamp,seconds,frame_number,track_id,class,confidence,direction
```

## Browser-Compatible Video

OpenCV writes MP4 output first. If `ffmpeg` is available on PATH, the app converts the result to H.264 with `yuv420p` pixel format for better browser playback. If `ffmpeg` is unavailable or conversion fails, the app keeps the OpenCV MP4 and shows a graceful warning.

## Current Scope And Future Features

This POC performs object counting. It does not claim defect detection unless you provide a trained defect model.

Future model-specific extensions can include:

- Defect Inspection
- PPE Detection
- Assembly Verification
- Missing Component Detection
