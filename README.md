# Bike Defect Detection - Two-Stage Pipeline

A deep learning system for detecting and localizing defects on bicycles using a two-stage approach: binary classification followed by anomaly localization.

## Overview

This project implements a **two-stage pipeline** for bike defect detection:

```
Input Image
     |
     v
+------------------+
|   STAGE 1        |
|   Binary         |
|   Classifier     |
+------------------+
     |
     v
  Intact? -----> YES -----> Output: INTACT
     |
     NO
     |
     v
+------------------+
|   STAGE 2        |
|   DRAEM          |
|   Localization   |
+------------------+
     |
     v
Output: DAMAGED + Heatmap showing defect location
```

## Stage 1: Binary Classification

**Purpose:** Classify bike images as either **INTACT** or **DAMAGED**

**Model:** EfficientNet-based classifier with custom head

**Key Features:**
- Mixup augmentation for regularization
- Label smoothing (0.1) to prevent overconfidence
- Focal Loss for handling class imbalance
- Weighted sampling to balance training data
- Detection threshold of 0.4 for damaged class

**Training Script:** `train_stage1_final.py`

```bash
python train_stage1_final.py \
    --config configs/stage1.yaml \
    --data_dir data/processed \
    --epochs 60 \
    --batch_size 16
```

**Output:** Binary prediction (Intact/Damaged) with confidence score

## Stage 2: DRAEM Anomaly Localization

**Purpose:** Localize the exact defect regions on damaged bikes

**Model:** DRAEM (Discriminatively trained Reconstruction Anomaly Embedding Model)

**Architecture:**
- **Reconstructive Sub-Network:** Encoder-decoder that learns to reconstruct normal bike images
- **Discriminative Sub-Network:** Segments anomalous regions by comparing input with reconstruction

**Key Features:**
- Trained only on intact/normal images
- Synthetic anomaly generation during training
- Multi-loss training: L2 reconstruction + SSIM + Focal segmentation loss
- Post-processing with morphological operations
- Bike mask segmentation to reduce background false positives

**Training Script:** `train_draem.py`

```bash
python train_draem.py \
    --config configs/draem_config.yaml \
    --intact_dir data/processed/intact \
    --epochs 100
```

**Inference Script:** `inference_draem.py`

```bash
python inference_draem.py \
    --model checkpoints/draem/draem_best.pth \
    --image test_image.jpg \
    --threshold 0.6
```

**Output:** Anomaly heatmap overlay showing defect locations

## Project Structure

```
bike_defect_detection/
|-- app.py                    # Streamlit web interface
|-- bike_pipeline.py          # Stage 1 inference pipeline
|-- inference_draem.py        # Stage 2 DRAEM inference
|-- train_stage1_final.py     # Stage 1 training script
|-- train_draem.py            # Stage 2 training script
|-- requirements.txt          # Python dependencies
|-- configs/                  # Configuration files
|-- src/
|   |-- models/
|   |   |-- classifier.py     # BikeClassifier model
|   |   |-- draem.py          # DRAEM model architecture
|   |-- data_preprocess/
|       |-- data_loader.py    # Data loading utilities
|       |-- anomaly_generator.py  # Synthetic anomaly generation
|-- checkpoints/              # Saved model weights
|-- data/                     # Training and test data
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Upload a bike image through the web interface to get:
- Stage 1: Damage classification result
- Stage 2: Defect localization heatmap (if damaged)

### Command Line Inference

**Stage 1 Only:**
```bash
python bike_pipeline.py \
    --image path/to/bike.jpg \
    --classifier checkpoints/stage1/best_model.pth
```

**Stage 2 (DRAEM):**
```bash
python inference_draem.py \
    --image path/to/bike.jpg \
    --model checkpoints/draem/draem_best.pth
```

## Data Preparation

Organize your data as follows:

```
data/
|-- processed/
|   |-- intact/      # Normal bike images
|   |-- damaged/     # Damaged bike images
```

The training scripts will automatically split data into train/validation/test sets (70/15/15).

## Model Performance

### Stage 1 Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix for both classes
- ROC-AUC curve

### Stage 2 Metrics
- Reconstruction loss (MSE + SSIM)
- Segmentation accuracy
- Anomaly localization IoU

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependencies

## License

This project is for educational and research purposes.