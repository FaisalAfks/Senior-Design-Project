# Face Recognition

End-to-end face security stack organised into three focused systems:

- `detection/` - BlazeFace model and helpers
- `recognition/` - MobileFaceNet, facebank tools, and recognition demos
- `antispoof/` - DeePixBiS model, dataset loaders, and liveness demos

The top-level `main.py` script unifies the stack, wiring up detection, recognition, and anti-spoofing in a single CLI.

## Layout
```
main.py                # unified CLI (image/video/webcam)
face_guardian_pipeline.py       # orchestrates detection + recognition + spoof

dataset/
    `- facebank/                # enrolled identities for recognition

detection/
    |- blazeface.py             # BlazeFace network definition
    |- blazeface_detector.py    # detector wrapper + alignment helpers (eyes/nose/mouth)
    `- blazeface_assets/        # pretrained weights + anchors

recognition/
    |- mobilefacenet.py         # MobileFaceNet backbone + facebank + landmark utilities
    |- take_picture.py          # capture facebank crops
    `- weights/                 # MobileFaceNet pretrained weights

antispoof/
    |- antispoof_model.py       # DeePixBiS (DenseNet-161 encoder)
    |- antispoof_dataset.py / antispoof_loss.py / antispoof_metrics.py / antispoof_trainer.py
    |- antispoof_train.py / antispoof_test.py
    `- weights/                 # DeePixBiS pretrained weights

requirements.txt                # consolidated dependency list
```

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate   # PowerShell (use bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Key scripts
- **Unified guard:** `python main.py --source 0`
- **Facebank capture:** `python recognition/take_picture.py -n Alice`
- **Anti-spoof webcam only:** `python antispoof/antispoof_test.py`
- **Update embeddings after adding images:**
  ```bash
  python main.py --source 0 --update
  ```

Common options: `--detector-thr`, identity threshold `-th`, spoof threshold `--spoof-thr`, `--tta` for flip augmentation, and `--disable-spoof` to skip liveness when needed.

## Notes
- BlazeFace alignment now focuses on the inner facial landmarks (eyes, nose, mouth) so profiles stay stable even when ears leave the frame.
- DeePixBiS loads via the modern `weights=` argument while remaining compatible with the original checkpoint.
- Each sub-package exposes its own demos and utilities, but everything can be orchestrated through `main.py` for a full pipeline run.





