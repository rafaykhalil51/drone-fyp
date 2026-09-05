# Accessory model directory

Put the trained accessory YOLO weights here:

```
models/accessory_best.pt
```

The application loads this file automatically at startup. No code change and
no UI setting is required — the path comes from `config.yaml`:

```yaml
accessories:
  enabled: true
  model_path: "models/accessory_best.pt"
  confidence: 0.35
  imgsz: 960
```

## Required classes

The model must be able to resolve all four classes:

| Required | Accepted labels in the trained model |
|---|---|
| `cap` | `cap`, `hat`, `baseball_cap`, `beanie`, `helmet`, `hardhat`, … |
| `mask` | `mask`, `face_mask`, `facemask`, `with_mask`, `surgical_mask`, `n95`, … |
| `glasses` | `glasses`, `eyeglasses`, `spectacles`, `sunglasses`, `goggles`, … |
| `headphones` | `headphones`, `headphone`, `headset`, `earphones`, `earbuds`, … |

The alias table lives in one place, `model_loader.py`, so no alias handling is
scattered through the codebase. Labels that describe *absence* — `without_mask`,
`no_helmet`, `mask_weared_incorrect` — are deliberately ignored and never
counted as someone wearing the accessory.

If a class cannot be resolved, the dashboard reports which ones are missing and
keeps accessory detection disabled rather than showing incomplete statistics.

## Without this file

Person detection, BoT-SORT tracking, unique person counting, the annotated
output video and CSV/JSON export all work normally. Only the four accessory
counts are unavailable; they are shown as `--` / "Model offline" and are never
estimated or fabricated.

A person detector alone cannot supply these classes: `yolov8n.pt` is a COCO
model and COCO contains no cap, mask, glasses, or headphones category.
