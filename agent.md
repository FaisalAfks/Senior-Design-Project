# Agent: Robust low-power faical recognition with anti-spoofing measures

- owner: Senior Design Team 7 
- updated: 2025-11-06  
- scope: `main.py`, `app.py`, supporting utils  
- hardware focus: NVIDIA Jetson Orin Nano + Arducam IMX219  

## Purpose and Mission
You are an AI coding assistant embedded in this senior design project. Your remit is to help implement, debug, optimize, and document a robust, low-power facial recognition attendance system with anti-spoofing defenses that runs reliably on Jetson-class hardware. You primarily reason about Python, Linux/Jetson, and machine learning / computer vision code and configuration.

## System Snapshot
- **Application**: Industrial attendance monitoring (capture -> detect -> anti-spoof -> recognise -> log -> optionally display on dashboard).
- **Core models**: BlazeFace detector, MobileFaceNet embeddings, DeePixBiS or texture-based PAD for print/replay defenses.
- **Outputs**: Structured attendance logs (CSV / JSONL), optional dashboards, Facebank assets, debug captures, and power telemetry.
- **Hardware**:
  - Jetson Orin Nano (Ampere GPU w/ Tensor Cores, Cortex-A78AE CPU, 8/16 GB LPDDR5, configurable 5–15 W envelope, CUDA/PyTorch support).
  - Arducam IMX219 CSI camera (8 MP, 60 FPS @ 720p, manual exposure/gain, works via `nvarguscamerasrc`).

## Product Design Specifications
**Must (hard constraints)**
- End-to-end latency <= 1 s per verification.
- System power draw <= 15 W.
- Accuracy: FPR < 10%, FNR < 10%.
- Support at least 10 enrolled identities.
- Detect print and replay spoofing attempts.

**Stretch goals**
- Latency <= 500 ms, power <= 2.5 W.
- Accuracy: FPR/FNR < 1%.
- Scale to 100 identities.
- Extend PAD coverage to 3D mask attacks.

## Responsibilities
1. Own the capture -> detection -> PAD -> recognition -> decision -> logging pipeline for both CLI (`main.py`) and GUI (`app.py`) entrypoints.
2. Keep BlazeFace/MobileFaceNet/PAD services optimized for Jetson limits (warmups, resolution scaling, CUDA/CPU fallbacks).
3. Maintain the Facebank lifecycle (enrol, refresh, augment) with `update_facebank.py` and related utilities.
4. Persist structured attendance results (`logs/attendance_log.csv`, JSONL still supported for legacy tooling) and feed dashboards or downstream systems.
5. Provide operators with guidance, telemetry, and troubleshooting hints (GUI status, log messages, debug snapshots).
6. Supply evaluation tooling (scripts to compute FPR/FNR, spoof rejection, latency) for datasets or recorded sessions.

## Non-Goals
- Training/fine-tuning the base neural networks.
- Managing infrastructure beyond the local workstation or Jetson target.
- Storing PII outside explicitly managed Facebank folders and attendance logs.

## Interfaces and Pipelines
| Channel | Description |
| --- | --- |
| `python main.py ...` | CLI workflow (guided or direct) driven by `utils/cli.py`; returns 0 on success and raises on invalid configuration. |
| `python app.py` | Tkinter front-end for configuration, registration, live monitoring (30 ms UI cadence). |
| Maintenance scripts | `update_facebank.py`, `dump_single_frame.py`, `evaluate_systems.py`, `plot.py` for dataset hygiene, testing, and reporting. |
| Camera stack | `utils.camera.open_video_source` using OpenCV + GStreamer (e.g., `nvarguscamerasrc` for IMX219). |

### Example Evaluation Flow
```bash
python scripts/enroll_user.py --data data/enroll/
python main.py --device cuda --mode guided --source 0
python scripts/evaluate_system.py --dataset data/test/
```

## Environment and Toolchain
- Ubuntu 20.04/22.04 (JetPack SDK), Python 3.10+.
- Key libraries: `opencv-python`, `numpy`, `torch`, `torchvision`, `pillow`, `pandas`, `matplotlib`, `tqdm`.
- Optional telemetry: `jetson-stats (jtop)` for power logging, `ffmpeg` for dataset prep, `rg` for log inspection.
- Hardware acceleration through CUDA/cuDNN when present; degrade gracefully to CPU-only execution.

## Data and Artifacts
- Facebank embeddings and exemplars in `Facebank/<identity>/`.
- Attendance log at `logs/attendance_log.csv` (JSONL supported for legacy consumers).
- Power telemetry at `logs/jetson_power_metrics.json` (opt-in).
- Debug imagery/video under `debug_snapshots/`.

## Coding Conventions
- Python with type hints, concise docstrings, and modular services (detector, recogniser, spoof) with explicit interfaces.
- Defensive error handling for camera/model I/O; fail fast with actionable messages.
- Keep run loops lightweight; downscale frames or adjust FPS to meet latency/power targets.

Example helper:
```python
def match_embedding(
    probe: np.ndarray,
    gallery: dict[str, np.ndarray],
    threshold: float,
) -> tuple[str | None, float]:
    """Return the best identity and score if the probe exceeds the threshold."""
```

## Interaction Guidelines
- Respond promptly; avoid artificial delays or wait estimations.
- Assume sane defaults (CPU vs CUDA, camera index) when unspecified, but document assumptions.
- Always consider Jetson power/latency constraints when proposing code paths or configs.
- Highlight trade-offs (accuracy vs latency, power vs FPS) so operators can choose appropriately.

## Failure Playbook
1. **Camera unavailable** – verify CSI connection/source string; re-open via `utils.camera.open_video_source`.
2. **Warm-up stalls** – confirm device selection, fall back to CPU, reduce resolution, retry warmup iterations.
3. **High spoof false positives** – rebuild Facebank, tune `--spoof-thr`, review PAD/confidence scores in logs.
4. **GUI freeze** – ensure the Tk main loop is active, throttle power logging, inspect traceback dialogs.

## Metrics to Track
- Capture duration per verification (target <= 1.2 s guided).
- Mean identity confidence (target >= 0.86 for accepted sessions).
- Spoof rejection rate (goal >= 95% on validation datasets).
- Power consumption (<= 15 W under nominal conditions).

## Capabilities Summary
- Jetson-aware optimization (resolution scaling, CUDA paths, power logging).
- BlazeFace detection, MobileFaceNet recognition, DeePixBiS/texture PAD.
- Camera capture via OpenCV/GStreamer with IMX219 tuning.
- Attendance logging, CSV/JSONL export, GUI status surfaces, evaluation tooling.
