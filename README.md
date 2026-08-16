# 2D Object Detection for Autonomous Driving using YOLO

**Course**: EEA-EV008 - Visual Perception and Planning for Autonomous Driving  
**Project**: 2D Object Detection on KITTI Dataset  
**Model**: YOLO11m (Latest Ultralytics implementation)  
**Dataset**: KITTI Vision Benchmark Suite  

---

## 📋 Project Overview

This project implements a state-of-the-art 2D object detection system for autonomous driving using the YOLO11 (You Only Look Once) model trained on the KITTI dataset. The system detects and localizes objects (cars, pedestrians, cyclists, etc.) in road scenes by predicting 2D bounding boxes and assigning object class labels.

### Key Objectives
- ✅ Develop and train a YOLO11m model on KITTI dataset
- ✅ Achieve competitive mAP scores (target: 70%+)
- ✅ Evaluate performance following KITTI benchmark protocol
- ✅ Analyze results across difficulty levels
- ✅ Provide qualitative visualizations
- ✅ Document methodology and findings in technical report

### Performance Targets
| Metric | Target |
|--------|--------|
| Overall mAP50 | 68-72% |
| Car AP50 | 75-80% |
| Pedestrian AP50 | 50-55% |
| Cyclist AP50 | 40-45% |
| Inference Speed | 4-5ms/image |

---

## 🗂️ Project Structure

```
Project/
├── README.md                          # This file
├── pyproject.toml                    # uv project dependencies
├── uv.lock                           # uv lockfile
├── requirements.txt                  # Optional compatibility dependency list
├── kitti.yaml                         # Local official KITTI dataset configuration
│
├── train_yolo_kitti.py               # Training script
├── inference_yolo_kitti.py           # Inference & visualization script
├── evaluate_kitti.py                 # KITTI evaluation script
│
├── data/
│   └── kitti/                        # Official KITTI dataset (manually downloaded)
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── runs/
│   ├── detect/
│   │   └── yolo11m_kitti_YYYYMMDD_HHMMSS/  # Training outputs
│   │       ├── weights/
│   │       │   ├── best.pt              # Best model
│   │       │   └── last.pt              # Last checkpoint
│   │       ├── results.csv              # Training metrics
│   │       └── plots/                   # Training visualizations
│   ├── inference_output/
│   │   ├── kitti_format/               # Predictions in KITTI format
│   │   └── visualizations/             # Result visualizations
│   └── evaluation_results/
│       └── results.json                # Evaluation metrics
│
├── results/                            # Final analysis results
└── visualizations/                     # Final visualizations
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone/Download this project
cd Project

# Install and lock dependencies with uv
uv sync

# Verify installation
uv run python -c "from ultralytics import YOLO; print('Ready to go!')"
```

### 2. Download KITTI Dataset

```bash
# Download official KITTI from the course PDF link (KITTI benchmark site)
# Extract images and labels to: data/kitti/images/ and data/kitti/labels/

# Generate local dataset YAML template once
uv run python train_yolo_kitti.py --init-data-config --data kitti.yaml
```

### 3. Train Model

```bash
# Train with default settings (YOLO11m, 100 epochs)
uv run python train_yolo_kitti.py

# Or customize parameters
uv run python train_yolo_kitti.py --model yolo11m.pt --epochs 100 --batch 16 --imgsz 640

# Use larger model for higher accuracy
uv run python train_yolo_kitti.py --model yolo11l.pt --epochs 100 --batch 8

# Check GPU usage
uv run python train_yolo_kitti.py --epochs 100 --device 0
```

**Expected training time**: ~3-5 hours on GPU

### 4. Run Inference

```bash
# Inference on validation images
uv run python inference_yolo_kitti.py \
    --model runs/detect/yolo11m_kitti_YYYYMMDD_HHMMSS/weights/best.pt \
    --source data/kitti/images/val \
    --conf 0.5 \
    --visualize

# Inference on single image
uv run python inference_yolo_kitti.py \
    --model runs/detect/yolo11m_kitti_YYYYMMDD_HHMMSS/weights/best.pt \
    --source image.jpg
```

### 5. Evaluate Results

```bash
# Compute AP following KITTI protocol
uv run python evaluate_kitti.py \
    --predictions runs/inference_output/kitti_format \
    --ground-truth data/kitti_official/label_2_val_split \
    --output runs/evaluation_full_local_kitti
```

---

## 📊 Detailed Usage Guide

### Training Script

```bash
usage: train_yolo_kitti.py [-h] [--model MODEL] [--epochs EPOCHS] [--batch BATCH] 
                            [--imgsz IMGSZ] [--device DEVICE] [--data DATA]
                            [--exp-name EXP_NAME] [--validate] [--export EXPORT]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model: yolo11n, yolo11m (default), yolo11l, yolo11x
  --epochs EPOCHS       Number of training epochs (default: 100)
  --batch BATCH         Batch size (default: 16, adjust for your GPU)
  --imgsz IMGSZ         Input image size (default: 640)
  --device DEVICE       GPU device ID (default: 0, use -1 for CPU)
  --data DATA           Dataset YAML file (default: kitti.yaml)
  --exp-name EXP_NAME   Experiment name for logging
  --validate            Run validation after training
  --export EXPORT       Export format: onnx, torchscript, tflite, pb, savedmodel
```

**Example training commands:**

```bash
# Standard training (YOLO11m, 100 epochs)
uv run python train_yolo_kitti.py --epochs 100

# High accuracy mode (YOLO11l)
uv run python train_yolo_kitti.py --model yolo11l.pt --epochs 100 --batch 8

# Fast training (YOLO11n, lower batch size for speed)
uv run python train_yolo_kitti.py --model yolo11n.pt --epochs 50 --batch 32

# Advanced: custom configuration
uv run python train_yolo_kitti.py \
    --model yolo11m.pt \
    --epochs 150 \
    --batch 16 \
    --imgsz 1024 \
    --device 0 \
    --validate \
    --export onnx
```

### Inference Script

```bash
usage: inference_yolo_kitti.py [-h] --model MODEL --source SOURCE [--conf CONF]
                               [--device DEVICE] [--save-txt]
                               [--visualize] [--output-dir OUTPUT_DIR]

required arguments:
  --model MODEL      Path to trained model (.pt)
  --source SOURCE    Image or directory path

optional arguments:
  --conf CONF        Confidence threshold (default: 0.5)
  --device DEVICE    GPU device ID (default: 0)
  --save-txt         Save predictions as text files
  --visualize        Save visualization images
  --output-dir       Output directory for predictions
```

**Example inference commands:**

```bash
# Inference with visualization
uv run python inference_yolo_kitti.py \
    --model runs/detect/yolo11m_kitti_20260403_120000/weights/best.pt \
    --source data/kitti/images/val \
    --conf 0.5 \
    --visualize

# Low confidence for recall
uv run python inference_yolo_kitti.py \
    --model best.pt \
    --source data/kitti/images/val \
    --conf 0.25 \
    --visualize

# High confidence for precision
uv run python inference_yolo_kitti.py \
    --model best.pt \
    --source data/kitti/images/val \
    --conf 0.75
```

### Evaluation Script

```bash
usage: evaluate_kitti.py [-h] --predictions PREDICTIONS 
                         --ground-truth GROUND_TRUTH [--output OUTPUT]

required arguments:
  --predictions PREDICTIONS    Directory with prediction files
  --ground-truth GROUND_TRUTH  Directory with ground truth files

optional arguments:
  --output OUTPUT              Output directory (default: runs/evaluation_results)
```

**Example evaluation:**

```bash
uv run python evaluate_kitti.py \
    --predictions runs/inference_output/kitti_format \
    --ground-truth data/kitti_official/label_2_val_split \
    --output runs/evaluation_full_local_kitti
```

---

## 📈 Training Workflow

### Step-by-Step Training Process

```
1. Setup Environment
   ├─ Install dependencies
   ├─ Verify GPU/CUDA
   └─ Download KITTI dataset

2. Configure Model
   ├─ Select YOLO variant (recommend: yolo11m)
   ├─ Set hyperparameters
   └─ Create kitti.yaml config

3. Train Model
   ├─ Start training script
   ├─ Monitor metrics:
   │  ├─ Training loss (box, cls, dfl)
   │  ├─ Validation mAP
   │  └─ Learning rate progress
   ├─ Early stopping if needed
   └─ Save best.pt

4. Validate Performance
   ├─ Run inference on val set
   ├─ Compute AP per class
   ├─ Check performance by difficulty
   └─ Analyze failure cases

5. Fine-tune (Optional)
   ├─ Adjust hyperparameters
   ├─ Retrain with modifications
   └─ Compare with baseline

6. Generate Report
   ├─ Compile results
   ├─ Create visualizations
   ├─ Write technical report
   └─ Prepare presentation
```

### Hyperparameter Tuning Strategy

**Key hyperparameters to tune:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| lr0 (initial lr) | 0.01 | 0.001-0.1 | Start high, reduce if unstable |
| lrf (final lr) | 0.01 | 0.001-0.1 | End learning rate |
| momentum | 0.937 | 0.8-0.99 | Momentum for SGD |
| weight_decay | 0.0005 | 0-0.01 | L2 regularization |
| batch | 16 | 8-64 | Larger batch = more stable |
| epochs | 100 | 50-200 | More epochs for better fit |
| warmup_epochs | 3 | 0-5 | Linear warmup before training |

**Tuning process:**
1. Start with defaults
2. Train for 10 epochs and check loss trends
3. If loss is stable, proceed to full training
4. Monitor val_loss for overfitting
5. Apply early stopping if val_loss increases

---

## 📊 Local evaluation results

These metrics are from the project’s local KITTI-style evaluator and reflect the measured outputs from this repository. They are not official KITTI leaderboard scores.

> Results were measured with this project’s local KITTI-style evaluator, not the official KITTI leaderboard evaluation.

### Moderate-difficulty comparison

| Model | Moderate AP (%) |
|-------|----------------:|
| YOLO11m | 94.66 |
| YOLO26s | 91.87 |
| YOLO26n | 85.24 |

### YOLO11m class-level results

| Class | AP (%) |
|-------|--------:|
| Car | 99.11 |
| Pedestrian | 89.70 |
| Cyclist | 95.18 |

### Evaluation context

- Dataset: KITTI-style local validation split used in this project
- Metric interpretation: local detection accuracy under the project evaluator
- Reporting standard: measured values from this repository’s saved evaluation outputs
- Not intended as: official KITTI leaderboard or benchmark publication

### Potential improvements

- **Model size**: explore larger variants if a different speed/accuracy tradeoff is required
- **Data augmentation**: improve robustness under occlusion and small-object cases
- **Fine-tuning**: adjust training configuration to further optimize the local validation split
- **Ensemble**: combine model variants if a small gain is needed for a specific deployment target

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'ultralytics'"
- **Solution**: `pip install ultralytics`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or image size
  ```bash
  uv run python train_yolo_kitti.py --batch 8 --imgsz 640
  ```

**Issue**: Very slow training
- **Solution**: Check GPU availability
  ```bash
  python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
  ```

**Issue**: Poor training metrics (loss not decreasing)
- **Solution**: 
  - Increase learning rate: `--lr0 0.05`
  - Reduce batch size for more gradient updates
  - Check if data is corrupted

**Issue**: Model converges too quickly (overfitting)
- **Solution**:
  - Reduce epochs with early stopping
  - Increase regularization: `--weight_decay 0.001`
  - Add more data augmentation

---

## 📝 Project Deliverables

### 1. Technical Report (IEEE Format)

**Sections to include:**

- **Abstract**: ~200 words summarizing approach and results
- **Introduction**: Problem statement, dataset overview
- **Methodology**: 
  - YOLO architecture overview
  - Training pipeline details
  - Hyperparameter settings
- **Experiments & Results**:
  - Performance metrics (mAP, by class, by difficulty)
  - Comparison with baselines
  - Inference speed analysis
- **Analysis**:
  - Strengths and limitations
  - Failure case analysis
  - Performance by scenario (occlusion, scale, etc.)
- **Conclusion**: Key findings and future work
- **Appendix**: Problems faced and solutions

### 2. GitHub Repository

**Required files:**
```
Repository/
├── README.md                # Project overview
├── requirements.txt         # Dependencies
├── *.py                    # Python scripts
├── data/                   # Dataset (or instructions)
├── docs/                   # Documentation
└── results/                # Final results and plots
```

### 3. Presentation

- Brief overview of approach
- Key results and metrics
- Visualizations of detected objects
- Discussion of challenges and solutions

---

## 📚 References

### Official Resources
- **Ultralytics Documentation**: https://docs.ultralytics.com/
- **YOLO GitHub**: https://github.com/ultralytics/ultralytics
- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/

### Research Papers
- **YOLOv10**: https://arxiv.org/abs/2405.14458
- **YOLOv9**: https://arxiv.org/abs/2402.13616
- **Object Detection Survey**: https://arxiv.org/abs/1809.02165
- **KITTI Benchmark**: https://arxiv.org/abs/1504.00325

### Useful External Links
- [YOLO Docs - KITTI Integration](https://docs.ultralytics.com/datasets/detect/kitti/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📄 License & Attribution

This project is part of the EEA-EV008 course at [Your University].

**Citation for KITTI Dataset:**
```
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR},
  year = {2012}
}
```

---

## 📞 Contact & Support

For issues or questions:
- Check the troubleshooting section
- Review IMPLEMENTATION_PLAN.md for detailed guidance
- Consult official Ultralytics documentation

---

**Last Updated**: April 3, 2026  
**Status**: Ready for Training  
**Estimated Completion**: 2-3 weeks

