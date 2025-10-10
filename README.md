# Face Guardian Toolkit

End-to-end face security stack organised into three focused systems that mirror the original research repositories:

- `BlazeFace/` – face detection, alignment helpers, demos, and notebooks
- `MobileFaceNet/` – recognition backbone, facebank utilities, tutorials, and demos
- `DeePixBis/` – anti-spoof model, unified training helpers, and evaluation scripts

The top-level `main.py` script now talks to each project through small service abstractions, so the legacy `face_pipeline.py` is no longer needed.

## Layout
```
main.py                        # unified CLI (image/video/webcam) using the new services
facebank/                      # enrolled identities stay in the project root

BlazeFace/
    |- api.py                  # service wrapper around BlazeFaceDetector
    |- detector.py             # detector wrapper + alignment helpers
    |- models/blazeface.py     # BlazeFace network definition
    |- Training/               # fine-tuning template & assets
    |- Testing/                # demos + sample images
    |- Weights/                # pretrained weights + anchors
    `- Notebooks/              # upstream notebooks (anchors, inference, conversion)

MobileFaceNet/
    |- api.py                  # MobileFaceNetService + facebank integration
    |- models/mobilefacenet.py # backbone, ArcFace head, facebank helpers
    |- Training/               # upstream training scripts & datasets helpers
    |- Testing/                # demos (camera/video/MTCNN, face capture, evaluation)
    |- Weights/                # pretrained MobileFaceNet checkpoint
    `- Notebooks/              # tutorial notebooks from the original repo

DeePixBis/
    |- api.py                  # DeePixBiSService for batched inference
    |- core.py                 # model, dataset, loss, metrics, trainer (consolidated)
    |- Training/train.py       # CLI training entry point
    |- Testing/test.py         # CLI evaluation entry point
    |- Weights/                # DeePixBiS pretrained checkpoint
    `- data/                   # CSV metadata + sample assets

requirements.txt               # consolidated dependency list
```

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate   # PowerShell (use bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Key scripts
- **Identity gate (alignment + 1s verification):** `python main.py --source 0`
- **Facebank capture:** `python MobileFaceNet/Testing/take_picture.py -n Alice`
- **Rebuild facebank from CLI:** `python MobileFaceNet/utils/facebank.py --rebuild`
- **Anti-spoof eval only:** `python DeePixBis/Testing/test.py --csv DeePixBis/data/val_metadata.csv --weights DeePixBis/Weights/DeePixBiS.pth`
- **DeePixBiS fine-tuning template:** `python DeePixBis/Training/train.py --train-csv ... --val-csv ...`
- **Legacy unified demo:** `python pipeline_demo.py --source 0` (original pipeline mode)

Common `main.py` options: `--detector-thr`, identity threshold `-th`, spoof threshold `--spoof-thr`, `--tta` for flip augmentation, `--update` to rebuild facebank, and `--disable-spoof` to skip liveness when needed.

## Notes
- BlazeFace notebooks, weights, and demos are grouped under `BlazeFace/` to match the upstream project naming.
- MobileFaceNet scripts now resolve imports via the package layout; demos load weights and facebank from the new paths automatically.
- DeePixBiS logic (dataset, loss, metrics, trainer) is consolidated into `core.py`, reducing script sprawl while keeping CLI entry points for training and testing.
- The `facebank/` directory remains at the project root so previously enrolled identities continue to work.
- Running `main.py` captures ~1 second of verified frames after alignment and appends the decision to `attendance_results.jsonl` for downstream attendance tooling.

