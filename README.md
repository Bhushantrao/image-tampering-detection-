# ELA-Enhanced Vision Transformer for Image Manipulation Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Django 5+](https://img.shields.io/badge/Django-5.0%2B-green.svg)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, end-to-end digital image forensics system designed to detect and localize digital image manipulations (splicing, copy-move, and object insertion) using a **Dual-Domain Hybrid Framework** combining **Error Level Analysis (ELA)**, **Vision Transformers (ViT)**, and **Explainable AI (Grad-CAM)**.

---

## 🔍 Key Architectural Innovations

1. **Dual-Domain Hybrid Ingestion**:
   * **Error Level Analysis (ELA)**: Computes compression artifact discrepancies at JPEG quality level 90 to expose quantization inconsistencies.
   * **RGB Semantic Projection**: Preserves natural RGB textures and edges for the Vision Transformer to avoid the semantic data loss common in pure ELA models.
2. **Vision Transformer (ViT-Base-Patch16-224)**:
   * Uses Multi-Head Self-Attention (12 heads, 12 blocks) to model long-range spatial dependencies across $16 \times 16$ image patches.
3. **Self-Attention Explainability (Grad-CAM)**:
   * Extracts activation gradients from Block 11 LayerNorm to produce physical boundary heatmaps of the manipulated zones.
4. **Gaussian Smoothing & Dynamic Filtering**:
   * Applies a $9 \times 9$ Gaussian filter ($\sigma = 1.5$) with a dynamic $60\%$ peak activation threshold ($\tau = 0.60 \times \max(\text{CAM})$) and suppresses token boundary leaks ($W, H \ge 150\text{ px}$).
5. **Precise Coordinate Extraction**:
   * Automatically isolates bounding boxes $(X, Y, \text{Width}, \text{Height})$ and computes exact tampered surface area percentages.
6. **Interactive Forensic Web Portal**:
   * Dynamic Alpha-blending slider to overlay self-attention heatmaps directly onto the original image.
   * Tabulated coordinates metadata inspector.
   * Live real-time Chart.js analytics dashboard.
   * Formal, print-ready forensic audit certificate export.

---

## 📊 Benchmark Performance (CASIA v2.0)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **83.38%** |
| **Precision** | **91.19%** |
| **Recall (Sensitivity)** | **87.19%** |
| **Specificity** | **84.50%** |
| **F1-Score** | **89.14%** |
| **AUC-ROC** | **0.9142** |
| **Localization IoU** | **0.95** |
| **Localization Dice Coefficient** | **0.97** |

---

## 📁 Project Structure

```text
imagetamperingdetection/
├── ImageTamperingDetection/    # Django project settings and routing
├── detector/                   # Core application
│   ├── ml/                     # Model architecture and training pipeline
│   │   ├── models/             # ViT model and training scripts
│   │   └── evaluation/         # Metrics and plotting routines
│   ├── templates/              # HTML5 UI templates (upload, result, analytics)
│   ├── models.py               # SQLite database models
│   ├── predict.py              # Dual-domain inference and Grad-CAM extraction
│   ├── utils.py                # ELA generation and image processing helpers
│   └── views.py                # Django views and API controllers
├── results/                    # Generated ROC, confusion matrix, and loss curves
├── best_vit_model.pth          # Trained PyTorch ViT weights checkpoint
├── manage.py                   # Django management script
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Bhushantrao/image-tampering-detection-.git
cd image-tampering-detection-
```

### 2. Create and Activate a Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Database Migrations
```bash
python manage.py migrate
```

### 5. Launch the Development Server
```bash
python manage.py runserver 127.0.0.1:8000
```
Open your browser and navigate to **`http://127.0.0.1:8000/`**.

---

## 🔬 Model Weights
The Vision Transformer model uses `vit_base_patch16_224` pretrained on ImageNet-21k and fine-tuned on the CASIA v2.0 dataset. The weights file `best_vit_model.pth` is tracked using **Git LFS** (Large File Storage).

To pull the weights file when cloning:
```bash
git lfs install
git lfs pull
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
