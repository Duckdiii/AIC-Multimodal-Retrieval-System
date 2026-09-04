# AIC AI Challenge

Desktop keyframe retrieval system for the AIC competition workflow. The current competition path uses a BTC CLIP ViT-B/32 unified index and exports real video frame IDs.

## Current Workflow

```text
Dataset / BTC features
-> Build System
-> .rvdb
-> Smart Load
-> Query
-> CLIP text encoding
-> FAISS retrieval
-> Top-K results
-> Nearby Frames
-> CSV export
```

The competition GUI fixes retrieval mode to `clip_only`. A BTC index selects `openai/clip-vit-base-patch32` as the active query encoder, matching the 512-dimensional FAISS vectors stored in the index.

## Setup

Python 3.11 is the tested version. Python 3.10-3.11 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python gui.py
```

For CUDA acceleration, install the PyTorch wheel appropriate for the local CUDA version before installing the remaining requirements. The repository does not pin a machine-specific CUDA wheel source.

## Smart Load

1. Start the application with `python gui.py`.
2. Open **System Info** and choose **Smart Load**.
3. Select `output_keyframes/AIC2026_BTC_CLIP32_strict.rvdb`.
4. Confirm that the index is loaded and the active model is CLIP B/32.
5. Search, inspect results or Nearby Frames, add frames to the CSV list, and export.

The `.rvdb` file is a generated/external retrieval artifact. It is intentionally excluded from Git and must be obtained separately from the team or rebuilt from the external dataset artifacts.

## External Data

The following are not stored in Git:

- `Keyframes/`
- `clip_features/`
- `map-keyframes-aic25-b1/`
- `media-info-aic25-b1/`
- `output_keyframes/AIC2026_BTC_CLIP32_strict.rvdb`

Teammates must obtain these artifacts separately. Do not commit API credentials, local model caches, generated indexes, logs, exports, or runtime databases.

## Expected Data Layout

```text
AIC2026_AI-Challenge/
|-- gui.py
|-- config.json
|-- requirements.txt
|-- Keyframes/
|   |-- L21_V001/
|   |   |-- 001.jpg
|   |   `-- ...
|   `-- ...
|-- clip_features/
|   `-- clip-features-32/
|       |-- L21_V001.npy
|       `-- ...
|-- map-keyframes-aic25-b1/
|   `-- map-keyframes/
|       |-- L21_V001.csv
|       `-- ...
`-- output_keyframes/
    `-- AIC2026_BTC_CLIP32_strict.rvdb
```

## Production Index

The production index is `AIC2026_BTC_CLIP32_strict.rvdb`. It contains the unified retrieval vectors and metadata used by Smart Load. Search can use the production index without re-encoding JPG files. Building a replacement BTC index requires the BTC `.npy` features, strict keyframe mapping CSVs, and keyframe images.

See [DATA_ARTIFACTS.md](DATA_ARTIFACTS.md) for the artifact manifest and checksum.
