# YOLOv12 Flask Object Detection — Deep Dive Guide

This guide explains the entire project—how it works, how to use it, and how to extend it. It’s written for both non-technical and technical readers and grounded in the actual code in this repository.

- App entrypoint: `app.py`
- Core model utilities: `yolo_utils.py`
- Configuration: `config.py`
- Frontend (UI): `templates/index.html`, `static/js/main.js`, `static/css/style.css`
- Assets and outputs: `uploads/`

---

## 1) Non-Technical Overview

- The app detects objects using a computer vision model (YOLOv12) in three ways:
  - Webcam stream (real-time)
  - Uploaded image
  - Uploaded video
- When running, the app shows annotated visuals with bounding boxes and labels.
- For counting, the app reports peak simultaneous objects (the maximum number seen in any single frame) to avoid inflated counts from frame-by-frame accumulation.
- The app is a website built with Flask (Python). You open it in a browser, click a tab (Webcam/Image/Video), and see results and summaries.

---

## 2) Quick Start

1) Install dependencies from `requirements.txt`.

2) Ensure the YOLOv12 model is present at the path in `config.py` (`MODEL_PATH = 'yolo12n.pt'`).

3) Run the app:

```bash
python app.py
```

4) Open your browser at http://127.0.0.1:5000 and use the tabs to analyze webcam, image, or video.

Outputs (e.g., annotated images and video summaries) are saved in `uploads/`.

---

## 3) System Architecture

- Frontend (browser): `templates/index.html` + `static/js/main.js` (+ `static/css/style.css`)
  - Tabs for Webcam/Image/Video.
  - Buttons to start webcam, upload files, and display results and charts.
- Backend (Flask): `app.py`
  - Routes for serving the page, receiving uploads, streaming webcam/video frames, and returning summaries.
- Model & Processing: `yolo_utils.py`
  - Loads YOLOv12 model (`YOLO(config.MODEL_PATH)`).
  - Processes frames with `process_frame()`.
  - Real-time thread for webcam: `WebcamStream`.
  - Video analysis with peak counting and summary persistence.
- Configuration: `config.py`
  - Paths and thresholds for detection and I/O.

Data flow example (video): Browser uploads video -> Flask saves file -> Browser streams processed frames from `/video_feed/<filename>` -> When done, frontend polls `/video_summary/<filename>` for the JSON summary saved by `yolo_utils.analyze_video()`.

---

## 4) Configuration — `config.py`

```python
# config.py
MODEL_PATH = 'yolo12n.pt'         # YOLOv12 model weights path
CONFIDENCE_THRESHOLD = 0.5        # Object detection confidence threshold
IOU_THRESHOLD = 0.4               # IoU for non-max suppression
WEBCAM_SOURCE = 0                 # Default camera index
UPLOADS_FOLDER = 'uploads'        # Where uploads and outputs are stored
```

Guidance:
- MODEL_PATH: Replace with custom model weights to detect different classes.
- CONFIDENCE_THRESHOLD: Increase for stricter detections; decrease for more detections.
- IOU_THRESHOLD: Adjust to control suppression of overlapping boxes.
- WEBCAM_SOURCE: Set to a different camera index or RTSP/HTTP stream (advanced cameras require different `cv2.VideoCapture` flags).
- UPLOADS_FOLDER: Ensure this folder exists and is writeable. `app.py` creates it if missing.

Scaling knobs (where to extend):
- If you need environment overrides (e.g., production), add `os.getenv` readers in `config.py` and use them across the app.
- If you need performance switches (frame resize, skip rate, encoding quality, FPS limit), move those constants into `config.py` (currently parameterized inside `yolo_utils.analyze_video`).

---

## 5) Backend — `app.py` routes and logic

File: `app.py`

- `index()` — `GET /`
  - Renders `templates/index.html`.

- `start_webcam()` — `POST /start_webcam`
  - Calls `yolo_utils.start_webcam_stream()`.
  - Returns `{ status: 'Webcam stream started' }`.
  - Note: The current frontend starts the webcam stream by hitting `/video_stream` directly, which auto-starts the webcam if needed.

- `stop_webcam()` — `POST /stop_webcam`
  - If `yolo_utils.webcam_stream` exists, obtains a session summary via `get_summary()`.
  - Stops the webcam stream via `yolo_utils.stop_webcam_stream()`.
  - Returns `{ status, summary }` where `summary` includes peak counts, session duration, and frames processed.

- `video_stream()` — `GET /video_stream`
  - Returns a multipart MJPEG stream from `yolo_utils.generate_frames()` for the webcam.
  - Automatically starts the webcam stream if it’s not running.

- `upload_image()` — `POST /upload_image`
  - Accepts an image file.
  - Saves to `uploads/`.
  - Calls `yolo_utils.analyze_image()` to get an annotated frame and summary.
  - Saves an annotated image (`uploads/annotated_<original>`), returns its URL and the summary.

- `upload_video()` — `POST /upload_video`
  - Accepts a video file.
  - Saves to `uploads/` and returns `{ video_path: <filename> }`.

- `video_feed(filename)` — `GET /video_feed/<filename>`
  - Streams a processed video (multipart MJPEG) using `yolo_utils.analyze_video(video_path)`.

- `video_summary(filename)` — `GET /video_summary/<filename>`
  - Returns the final JSON summary saved by `yolo_utils.analyze_video()`.

- `uploads/<filename>` — `GET /uploads/<filename>`
  - Serves saved uploads and generated annotated assets.

- App shutdown hook: `atexit.register(yolo_utils.stop_webcam_stream)` ensures webcam resources are released when the process exits.

---

## 6) Model & Processing — `yolo_utils.py`

File: `yolo_utils.py`

### 6.1 Model loading
```python
model = YOLO(config.MODEL_PATH)
```
- Loads the YOLOv12 model once at import time.
- `model.names` maps class indices to human-readable labels.

### 6.2 Frame processing — `process_frame(frame)`
```python
def process_frame(frame):
    results = model(frame, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD)
    annotated_frame = results[0].plot()  # BGR np.array
    detected_objects = []
    for result in results[0].boxes:
        cls = int(result.cls)
        label = model.names[cls]
        detected_objects.append(label)
    summary = { 'counts': dict(Counter(detected_objects)), 'processing_time': ... }
    return annotated_frame, summary
```
- Runs a single inference.
- Returns the annotated image and a summary that includes per-frame counts and timing.

### 6.3 Real-time webcam — `class WebcamStream`

- Starts a background thread reading frames from `cv2.VideoCapture(config.WEBCAM_SOURCE, cv2.CAP_DSHOW)`.
- For each frame, calls `process_frame()`, stores latest annotated frame for streaming, and appends per-frame counts to `detected_objects_history` for later summarization.
- Key methods:
  - `_update()`: Thread loop reading frames and processing detections.
  - `get_frame()`: Returns the latest annotated frame encoded as JPEG bytes.
  - `get_summary()`: Returns peak counts across all frames, session duration, and total frames processed.
  - `stop()`: Stops the thread and releases the camera.

Global webcam helpers:
- `start_webcam_stream()` — Creates the singleton `webcam_stream` if missing/stopped.
- `stop_webcam_stream()` — Stops and clears the global instance.
- `generate_frames()` — Yields multipart JPEG chunks for Flask streaming. If stream is not running, starts it.

### 6.4 Image analysis — `analyze_image(image_path)`
- Loads the image via `cv2.imread()`.
- Calls `process_frame()` and returns `(annotated_frame, summary)`.

### 6.5 Video analysis and peak counting — `analyze_video(video_path, frame_skip=5)`

Performance and counting logic:
- Opens video via `cv2.VideoCapture`.
- Resizes large frames to max width 640px to speed up inference while preserving aspect ratio.
- Processes every Nth frame (`frame_skip=5` by default) to reduce compute; non-processed frames reuse last annotated frame for smooth playback.
- Encodes JPEG at quality 70 for faster streaming.
- Adds a small delay (`time.sleep(0.03)`) to cap stream around 30 FPS.
- Collects per-processed-frame counts in `frame_detections`.
- At the end, computes peak counts: for each class, the maximum number detected in any frame. This avoids inflated totals typical of per-frame accumulation.
- Saves a JSON summary to `uploads/<video_basename>_summary.json` with:
  - `counts`: peak counts per class
  - `total_processing_time`, `total_frames`, `processed_frames`, `fps`

---

## 7) Frontend — `templates/index.html` and `static/js/main.js`

### 7.1 UI structure — `templates/index.html`
- Three tabs: `Webcam` (`#webcam`), `Image` (`#image`), `Video` (`#video`).
- Webcam section:
  - `#start-webcam-btn` to start streaming.
  - `#webcam-feed` shows the MJPEG stream from `/video_stream`.
  - `#close-webcam-btn` stops the stream by calling `/stop_webcam` and then displays a session summary.
  - `#webcam-result-container`, `#webcam-summary`, `#webcam-chart` show session analytics.
- Image section:
  - Upload input `#image-upload` and analyze button `#image-analyze-btn`.
  - `#annotated-image` and `#image-summary`, `#image-chart` display results.
- Video section:
  - Upload input `#video-upload` and analyze button `#video-analyze-btn`.
  - `#video-feed-display` streams processed frames from `/video_feed/<filename>`.
  - `#video-summary`, `#video-chart` populate once summary JSON becomes available.

### 7.2 Interactions — `static/js/main.js`

- `openTab(evt, tabName)` — Shows selected tab and marks tab button active.

- Image flow:
  - On `#image-analyze-btn` click, POST to `/upload_image` with the file.
  - On success, display `annotated_image_url` in `#annotated-image` and summary in `#image-summary`; render chart via `renderChart('image-chart', counts)`.

- Video flow:
  - On `#video-analyze-btn` click, POST to `/upload_video` with the file.
  - Set `#video-feed-display.src = /video_feed/<filename>` to stream processed frames.
  - Start polling `/video_summary/<filename>` via `pollForSummary()`. When available, render summary + chart.

- Webcam flow:
  - On `#start-webcam-btn` click: set `#webcam-feed.src = /video_stream?t=<time>` to start stream; show close button.
  - On `#close-webcam-btn` click: clear `#webcam-feed.src`, POST `/stop_webcam`, hide feed, remove fullscreen state, then display session summary via `displayWebcamSummary()` with peak counts, duration, and frame count.

- Charting:
  - `renderChart(canvasId, data)` renders a bar chart using Chart.js with keys as labels and values as counts.

- Summary formatting:
  - `formatSummary(summary, type)` renders HTML snippets for image/video summaries. For webcam summary, a dedicated `displayWebcamSummary(summary)` is used to include duration and frames processed.

---

## 8) Counting Strategy: Peak Simultaneous Objects

Why peak instead of cumulative?
- Per-frame accumulation inflates counts (e.g., 1 person visible for 200 frames -> 200 people). Peak reports the maximum simultaneous number seen at once, which better answers “how many were present at the same time.”

Where implemented:
- Webcam: `WebcamStream.get_summary()` computes max per class across all frames.
- Video: `analyze_video()` computes max per class across processed frames before saving the final summary JSON.

---

## 9) Performance Optimizations

- Frame resizing: Max width 640px before inference for faster processing.
- Frame skipping: Process every 5th frame by default for video; adjust as needed.
- JPEG quality: Encode annotated frames at quality 70 to reduce bandwidth.
- Stream pacing: Add ~30 FPS cap to avoid overwhelming the browser.
- Threaded webcam: Background thread ensures responsive UI during real-time inference.

Tuning suggestions:
- If your machine/GPU is powerful, increase frame size, reduce `frame_skip`, or increase JPEG quality.
- If you experience lag, increase `frame_skip` or reduce frame size further.
- Consider moving these knobs into `config.py` for centralized control.

---

## 10) Extensibility and Scaling

- Custom models: Update `config.MODEL_PATH` to a custom YOLO model trained on your classes.
- Additional inputs: Add new routes for RTSP/HTTP camera streams (configure `cv2.VideoCapture` accordingly). Add a new tab and similar JS flow.
- WebSocket streaming: For lower latency and bidirectional updates, consider a Socket/WebSocket solution instead of MJPEG.
- Multi-camera: Run multiple `WebcamStream` instances keyed by camera ID and expose routes like `/video_stream/<camera_id>`.
- GPU acceleration: Ensure proper CUDA/cuDNN setup. Ultralytics YOLO uses GPU automatically when available; you can also specify device in model call if needed.
- Batch processing: For offline video/image batches, add a background job queue (e.g., RQ/Celery) and a results API.
- Cloud deployment: Containerize (Docker), store uploads in S3 or similar, and serve static assets via CDN.
- CI/CD & testing: Add unit tests for `process_frame()` with sample images and mock model; use GitHub Actions to lint/test on push.

---

## 11) Deployment Notes

- Development: `python app.py` runs Flask with `debug=True`. For production, run behind a WSGI server (e.g., Gunicorn) and reverse proxy (nginx), and disable debug.
- Dependencies: See `requirements.txt`.
- Model file: Ensure `yolo12n.pt` is available at runtime (bundle or mount volume).
- File permissions: The app must be able to write to `uploads/`.

---

## 12) Troubleshooting

- Blank webcam feed:
  - Confirm camera is not in use by other apps and `config.WEBCAM_SOURCE` is correct.
  - Check console logs and server logs for errors; ensure stream MIME is `multipart/x-mixed-replace`.
- No detections:
  - Lower `CONFIDENCE_THRESHOLD` in `config.py`.
  - Verify `MODEL_PATH` points to correct weights.
- Slow video streaming:
  - Increase `frame_skip` in `analyze_video()`.
  - Lower JPEG quality from 70 or reduce resize target below 640.
- Summary not appearing for video:
  - Ensure polling endpoint `/video_summary/<filename>` is reachable and JSON file is being saved in `uploads/`.
- Crashes on startup:
  - Ensure Python version compatibility and that model dependencies are installed (`ultralytics`, `opencv-python-headless`, `numpy`).

---

## 13) API Reference (Concise)

- `GET /` → HTML UI
- `POST /start_webcam` → Start webcam (optional flow; stream auto-starts on `/video_stream`)
- `POST /stop_webcam` → Stop webcam and return session summary
- `GET /video_stream` → MJPEG webcam stream
- `POST /upload_image` → Upload image; returns `{ annotated_image_url, summary }`
- `POST /upload_video` → Upload video; returns `{ video_path }`
- `GET /video_feed/<filename>` → MJPEG video stream with detections
- `GET /video_summary/<filename>` → Final JSON summary for the video
- `GET /uploads/<filename>` → Serve uploaded/annotated files

---

## 14) File-by-File Map

- `app.py` — Flask routes, upload handling, and stream endpoints; ensures `uploads/` exists; exits cleanly.
- `yolo_utils.py` — YOLO model, frame processing, webcam stream thread, image/video analysis, peak counting, JSON summary writing.
- `config.py` — Central toggles for weights, thresholds, camera index, uploads folder.
- `templates/index.html` — UI layout with tabs for webcam, image, and video.
- `static/js/main.js` — Event handlers for uploads, streaming, polling, summaries, and chart rendering.
- `static/css/style.css` — Visual styling and responsive layout.
- `uploads/` — User uploads, annotated images, and video JSON summaries.

---

## 15) Future Enhancements

- Real-time alerts (e.g., send notification when a target class appears above a threshold).
- Region-of-Interest (ROI) counting and heatmaps.
- Tracking across frames (e.g., DeepSORT) for consistent IDs instead of pure per-frame detections.
- WebSocket-based low-latency streaming and event updates.
- Admin dashboard to manage multiple cameras and historical analytics.

---

## 16) Maintenance Checklist

- Keep `ultralytics` updated, but test before upgrading.
- Monitor disk usage in `uploads/` and clean up periodically.
- Validate file types on upload for security.
- Log errors and user actions for troubleshooting.

---

This document reflects the current codebase and is meant to be directly actionable for understanding, maintaining, and extending the project.
