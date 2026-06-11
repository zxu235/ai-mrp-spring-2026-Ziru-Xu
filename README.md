# Traffic Sign Detection Using YOLO26
### Mini Research Project — Ziru Xu | June 2026

---

## Overview

This project fine-tunes **YOLO26n** (Ultralytics, 2026) on a 15-class traffic sign dataset to detect and localize traffic signs in real car-perspective dashcam footage. The full pipeline — from dataset download to ONNX export — is contained in a single Jupyter notebook.

---

## Results

| Metric | Value |
|--------|-------|
| Precision | 82.5 % |
| Recall | 72.8 % |
| mAP @ 0.50 | 82.4 % |
| mAP @ 0.50:0.95 | 72.0 % |
| Epochs | 10 |
| Model | YOLO26n (2.4M params) |

---

## Project Structure

```
MRP/
├── MRP-YOLO.ipynb                      # Main notebook (full pipeline)
├── MRP Presentation_Ziru_Xu.pdf   # Presentation slides
├── Data/
│   └── video.mp4                       # Real dashcam footage (416×416, 30fps, 17s)
├── Self-Driving-Cars-6/                # Roboflow dataset (auto-downloaded)
│   ├── train/images/   (3,530 images)
│   ├── valid/images/   (801 images)
│   ├── test/images/    (638 images)
│   └── data.yaml
└── runs/detect/
    └── traffic_signs/                  # Training outputs
        ├── weights/
        │   ├── best.pt                 # Best checkpoint
        │   └── last.pt
        └── results.csv                 # Per-epoch metrics
```

---

## Dataset

**Source:** [Roboflow — Self-Driving Cars v6](https://roboflow.com)  
**Classes (15):** Green Light, Red Light, Stop, Speed Limit 10/20/30/40/50/60/70/80/90/100/110/120  
**Format:** YOLOv8 (bounding boxes, one `.txt` label per image)

| Split | Images |
|-------|--------|
| Train | 3,530 |
| Valid | 801 |
| Test | 638 |

---

## Model

**YOLO26n** — the nano variant of the YOLO26 family (Ultralytics, June 2026).

| Property | Value |
|----------|-------|
| Parameters | 2.4 M |
| GFLOPs | 5.4 |
| COCO mAP (pretrained) | 40.9 |
| Inference (CPU ONNX) | 43% faster than YOLO11n |

Key innovations over previous YOLO versions:
- **NMS-free** end-to-end inference (one-to-one detection head)
- **DFL-free** regression (lighter bounding-box branch)
- **MuSGD** optimizer (SGD + Muon hybrid)
- Progressive Loss + STAL for improved small-object recall

---

## Setup

```bash
# Install dependencies
pip install ultralytics roboflow

# Or with conda
conda activate <your-env>
pip install ultralytics roboflow
```

**macOS note:** If you see an OpenMP conflict, set:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**ffmpeg** is required for video display:
```bash
brew install ffmpeg   # macOS
```

---

## Running the Notebook

Open `MRP-YOLO.ipynb` and run cells top-to-bottom:

| Section | What it does |
|---------|-------------|
| 1. Dataset | Downloads Self-Driving Cars v6 from Roboflow, sets `DATASET_DIR` and `DATA_YAML` |
| 2. Baseline | Runs pretrained YOLO26n (COCO weights) on a sample image — no fine-tuning |
| 3. Fine-tuning | Trains YOLO26n for 10 epochs on the traffic signs dataset |
| 4. Evaluation | Loads `best.pt`, runs COCO-style validation, prints mAP/P/R |
| 5. Test Inference | 3×3 grid of annotated test images |
| 6. Video Inference | Runs detection on `Data/video.mp4`, displays annotated output |
| 7. Export | Exports `best.pt` to ONNX for deployment |

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 10 | |
| `batch` | -1 (auto) | Fills ~60% of available RAM |
| `optimizer` | auto → MuSGD | Selected automatically for YOLO26 |
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR multiplier (cosine decay) |
| `imgsz` | 640 | |
| `device` | cpu | |

---

## Limitations

- **CPU training** limits to 10 epochs; GPU training would push mAP@0.5 above 90%
- **Class imbalance** — speed limit classes dominate, reducing recall on Stop/Light classes
- **Small objects** — distant signs are frequently missed (Recall 72.8% vs Precision 82.5%)
- **Single domain** — trained on dashcam footage only; untested in rain/night conditions

---

## References

1. Jocher et al. (2026). *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.* arXiv:2606.03748
2. Redmon et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
3. Flores-Calero et al. (2024). *Traffic Sign Detection and Recognition Using YOLO.* Mathematics, 12(2), 297.
4. Jiang et al. (2024). *YOLO-CCA: A Context-Based Approach for Traffic Sign Detection.* arXiv:2412.04289.
