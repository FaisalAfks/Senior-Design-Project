Senior Design Project — Face Verification System
================================================

Overview
--------
This project implements a real‑time face verification workflow with optional on‑screen alignment guidance, identity recognition, and anti‑spoofing. It targets both standard desktops (CPU/GPU) and NVIDIA Jetson (Orin Nano et al.) devices.

Key Features
------------
- Guidance overlay to help subjects align before capture (guided mode)
- Real‑time face detection (BlazeFace), recognition (MobileFaceNet), and anti‑spoofing (DeePixBiS)
- Accurate timing with two evaluation modes:
  - Time budget (e.g., capture up to 1.0s)
  - Fixed frame count (e.g., capture exactly 30 frames)
- Attendance logging as JSONL with identity, spoof score, and capture duration
- Optional Jetson power telemetry logging via jetson‑stats (jtop)

Repository Layout
-----------------
- `main.py` — Entry point for the verification app (window UI, workflow)
- `utils/` — Helpers (CLI, camera, guidance overlay, verification loop, session control, power logging)
- `BlazeFace/` — Detector wrapper and model assets
- `MobileFaceNet/` — Embedding model, facebank utilities, and weights
- `DeePixBis/` — Anti‑spoofing model and weights
- `Facebank/` — Known identities (images/embeddings)
- `requirements.txt` — Python package requirements
- Additional tools: `update_facebank.py`, `dump_single_frame.py`, `evaluate_systems.py`, `plot.py`

Requirements
------------
- Python 3.10+
- OpenCV with GUI support (for windows and display)
- PyTorch (CPU or CUDA)
- For Jetson devices: install NVIDIA’s Jetson‑specific PyTorch wheels; install GStreamer support for CSI cameras

Install
-------
1) Create and activate a virtual environment (recommended).
2) Install PyTorch appropriate for your platform (CUDA on desktop GPU, NVIDIA wheel on Jetson).
3) Install the remaining dependencies:

    pip install -r requirements.txt

Model Assets and Facebank
-------------------------
- Ensure the following exist and contain weights/assets:
  - `MobileFaceNet/Weights/MobileFace_Net`
  - `BlazeFace/Weights/` (blazeface weights + anchors)
  - `DeePixBis/Weights/DeePixBiS.pth` (if anti‑spoofing enabled)
- Facebank directory structure (`Facebank/`) should contain one subfolder per identity with example face images.
- To rebuild the facebank embeddings from images, use:

    python main.py --update-facebank

or run `update_facebank.py` directly to regenerate embeddings using a dataset.

Running
-------
Basic CPU run with default webcam 0:

    python main.py --device cpu --mode guided --evaluation-mode time --evaluation-duration 1.0

Use CUDA (desktop GPU) if available:

    python main.py --device cuda:0 --mode guided

Jetson (CSI camera) example at 1280x720 @ 30 FPS:

    python main.py --device cuda --source csi://0?width=1280&height=720&fps=30 --camera-width 1280 --camera-height 720

Direct mode (skip guidance and start capture immediately):

    python main.py --mode direct

Evaluation Modes
----------------
- Time budget (default):

    --evaluation-mode time --evaluation-duration 1.0

  Captures frames for up to the specified number of seconds. The on‑screen progress is clamped to the budget, while the logged duration reflects true elapsed time.

- Fixed frame count:

    --evaluation-mode frames --evaluation-frames 30

  Captures exactly N frames (or until ESC). The logged duration reflects how long those frames took.

Window Controls
---------------
- SPACE: Continue to the next verification cycle
- ESC or q: Quit

Output and Logging
------------------
- Attendance results: appended to `logs/attendance_log.csv` (configurable via `--attendance-log`). Each row records
  `timestamp`, `source`, `recognized`, `identity`, `avg_identity_score`, `avg_spoof_score`, `is_real`, `accepted`,
  `frames_with_detections`, and `capture_duration`.

Jetson Power Logging (Optional)
-------------------------------
- Requires `jetson-stats` (jtop). Install on Jetson:

    sudo -H pip install jetson-stats

- Enable logging:

    python main.py --enable-power-log --power-interval 1.0 --power-log logs/jetson_power_metrics.json

- Each sample records both device-wide power and an estimate for this process:
  - `total_power_w` / `total_avg_power_w` come directly from the jtop `VDD_IN` rail.
  - `soc_power_w` sums CPU/GPU/SOC rails so you can see how much of the draw is going into computation.
  - `process` contains the tracked PID, CPU%, RAM/GPU usage, the proportional shares, and `estimated_power_w`, which allocates the SoC wattage using the process' CPU/GPU utilisation mix (method field explains the fallback path).

- You may also lock clocks for consistent latency (optional, requires sudo):

    sudo nvpmodel -m 0 && sudo jetson_clocks

Performance Tips
----------------
- First‑run CUDA warm‑up: the code proactively warms models to avoid the initial stall before the first capture.
- Reduce camera resolution or set `--frame-max-size` (e.g. `1280x720`) to limit processing cost.
- Tune thresholds: `--identity-thr` and `--spoof-thr` affect acceptance and may reduce extra work depending on usage.

Guidance Overlay
----------------
- Guided mode shows a square and prompts like “Align your face within the square”. When no face is detected, a “No face detected” notice appears; messages are positioned to avoid overlap.
- `--guidance-box-size` controls the square size (0 = auto). Center/size/rotation tolerances are configurable via CLI.

Troubleshooting
---------------
- No camera frames / black window:
  - Verify the `--source` and camera index; on Jetson, prefer `csi://` with explicit width/height.
  - Check that GStreamer is available when using CSI pipelines.
- CUDA first run is slow:
  - The warm‑up step should mitigate this. If you still see a long first pass, ensure your CUDA/PyTorch install matches the device.
- Few frames captured in 1s window:
  - Lower resolution, reduce model work (disable spoof with `--disable-spoof`), or use frames mode (`--evaluation-mode frames`).

Development Notes
-----------------
- Core capture/verification loop: `utils/verification.py` (see `run_verification_phase` and `evaluate_frame`).
- High‑level workflow: `main.py` sets up services, guidance, and session control.
- CLI options: `utils/cli.py` describes all flags and defaults.
- Services (detector/recogniser/spoof): constructed in `utils/services.py`.

License
-------
No license is provided in this repository. Add one if you plan to distribute or open‑source the project.
