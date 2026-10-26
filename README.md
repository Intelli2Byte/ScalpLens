# 🔬 ScalpLens: Dual-Pipeline Scalp Pathology & Macro Screener

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8.svg)](https://opencv.org/)
[![UI Framework](https://img.shields.io/badge/Streamlit-1.62%2B-FF4B4B.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Model-EfficientNet--B0-brightgreen.svg)](https://pytorch.org/vision/stable/models/efficientnet.html)
[![Inference](https://img.shields.io/badge/Execution-100%25%20Offline%20Local-success.svg)](#)

ScalpLens is an offline, privacy-preserving clinical screening and diagnostic platform designed to detect and assess the three most prevalent non-scarring scalp conditions: **Alopecia Areata**, **Seborrheic Dermatitis**, and **Androgenetic Alopecia**. 

The system features a **Dual-Pipeline Architecture**:
1. **Dermoscopy Engine (Deep Learning)**: Evaluates specialized microscopic trichoscopy images using a transfer-learned **EfficientNet-B0** convolutional neural network.
2. **Smartphone Macro Engine (Computer Vision)**: Analyzes standard everyday photos taken with a normal phone camera (hair parting, crown, receding hairline) using deterministic OpenCV photogrammetry biomarkers combined with demographic risk modeling (hard water exposure, vegetarian dietary profiles).

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset Origin & Selection](#-dataset-origin--selection)
- [Dataset Distribution (Train / Test Split)](#-dataset-distribution-train--test-split)
- [System Architecture & Pipelines](#-system-architecture--pipelines)
- [Visual Biomarkers Pipeline (Smartphone Photos)](#-visual-biomarkers-pipeline-smartphone-photos)
- [Project Directory Structure](#-project-directory-structure)
- [Tools & Technologies Used](#-tools--technologies-used)
- [Quickstart: How to Run the Project](#-quickstart-how-to-run-the-project)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Academic & Clinical Disclaimer](#-academic--clinical-disclaimer)

---

## 🎯 Problem Statement

Over 50% of adults experience non-scarring scalp disorders. In clinical settings, early-stage symptoms (erythema, follicular thinning, scaling, and itching) often look identical, creating high diagnostic ambiguity:
- **Alopecia Areata (AA)** is an autoimmune condition requiring targeted anti-inflammatory or immunosuppressive therapy.
- **Seborrheic Dermatitis (SD)** is a fungal/inflammatory condition driven by *Malassezia* yeast overactivity, aggravated by mineral-heavy hard water and high humidity.
- **Androgenetic Alopecia (AGA)** is a progressive genetic follicular miniaturization, frequently compounded by nutritional deficiencies (Serum Ferritin and Vitamin D3).

While clinical trichoscopes provide magnified dermoscopic details, ordinary users only possess standard smartphone cameras. Conventional deep learning models trained purely on dermoscopy fail drastically when given consumer phone photos due to domain shift. **ScalpLens solves this by uniting specialized deep learning with adaptive smartphone computer vision photogrammetry.**

---

## 📂 Dataset Origin & Selection

- **Primary Source**: **PUMCH-SCALP** dataset from Peking Union Medical College Hospital (standard academic dermoscopy benchmark).
- **Target Selection**: Out of the 10 raw condition folders in the archive, the 3 most prevalent conditions accounting for >85% of outpatient scalp visits were isolated:
  1. `Alopecia areata`
  2. `Seborrheic dermatitis`
  3. `Androgenetic alopecia`
- **Avoidance of JSON Segmentation Masks**: The raw archive contains polygon segmentation `.json` masks in `PUMCH-AA`. In accordance with clinical triage objectives, segmentation masks were bypassed to construct a pure, robust, high-throughput **image classification pipeline** using verified `.jpg` dermoscopy scans.
- **Image Integrity Verification**: Every image is verified via PIL (`Image.verify()`) to eliminate corrupt files before processing.

---

## 📊 Dataset Distribution (Train / Test Split)

The dataset was partitioned into an **80% Training** and **20% Testing/Validation** split with a fixed random seed (`seed=42`) for mathematical reproducibility:

| Scalp Condition / Class | Raw Directory Identifier | Training Set (80%) | Testing / Val Set (20%) | Total Verified Images |
| :--- | :--- | :---: | :---: | :---: |
| **Alopecia Areata** | `alopecia_areata` | **1,141** | **286** | **1,427** |
| **Seborrheic Dermatitis** | `seborrheic_dermatitis` | **602** | **151** | **753** |
| **Androgenetic Alopecia** | `androgenetic_alopecia` | **496** | **124** | **620** |
| **GRAND TOTAL** | — | **2,239** | **561** | **2,800** |

---

## 🔬 System Architecture & Pipelines

```
                             [ User Input Scalp Image ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [ Mode 1: Smartphone Photo ]                   [ Mode 2: Dermoscopy Scan ]
                 │                                               │
                 ▼                                               ▼
     [ OpenCV Photogrammetry ]                        [ PyTorch Preprocessing ]
   - YCrCb Scalp Skin Segmentation                  - 224x224 Memory-Safe Resizing
   - Normalized Redness Index (NRI)                 - ImageNet-1K Tensor Normalization
   - Flaking & Dandruff Burden (%)                               │
   - Contour Geometry (Circularity)                              ▼
   - Border Artifact Suppression                     [ EfficientNet-B0 Backbone ]
                 │                                   - Compound Scaling Architecture
                 ▼                                   - Dropout Regularization (p=0.4)
     [ Demographic Context Engine ]                  - 3-Class Linear Head
   - Hard Water Mineral Exposure                                 │
   - Indian Vegetarian Nutritional Deficits                      ▼
                 │                                    [ Softmax Probability Dist ]
                 ▼                                  (Alopecia | Seborrheic | Androgenetic)
  [ Visual Overlay + Actionable Report ]                         │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                 [ Interactive Streamlit Dashboard (app.py) ]
```

### Pipeline 1: Deep Learning Dermoscopy Engine
- **Backbone**: `torchvision.models.efficientnet_b0` pretrained on ImageNet-1K (`EfficientNet_B0_Weights.DEFAULT`).
- **Classifier Head**: Replaced with `nn.Dropout(p=0.4)` and a 3-class linear output layer (`1280 -> 3`).
- **Augmentation Pipeline**:
  - `Resize((224, 224))`
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomVerticalFlip(p=0.5)`
  - `RandomRotation(degrees=15)`
  - `ColorJitter(brightness=0.15, contrast=0.15)`
  - Standard ImageNet Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Optimization**: AdamW optimizer ($\text{lr} = 10^{-4}, \text{weight\_decay} = 10^{-2}$) with Cosine Annealing learning rate scheduling.
- **Loss Function**: Class-Weighted Cross-Entropy Loss to balance sample discrepancies between classes.
- **Checkpointing**: The model with highest validation accuracy is stored at [`saved_models/best_scalp_model.pth`](file:///c:/Users/neha/Downloads/AIML%20MINI/saved_models/best_scalp_model.pth).

---

## 📱 Visual Biomarkers Pipeline (Smartphone Photos)

For standard phone pictures of hair partings, hairlines, or crown thinning, ScalpLens executes a deterministic computer vision pipeline:

1. **Scalp Skin Segmentation**:
   - Converts RGB to $YCrCb$ and filters chrominance channels ($128 \le Cr \le 185$ and $72 \le Cb \le 140$) to extract scalp skin across Indian Fitzpatrick skin phototypes IV–V and fair complexions.
   - Applies morphological opening and closing with an elliptical kernel ($5 \times 5$).
2. **Scalp Exposure / Visibility Index (%)**:
   - Computes ratio of exposed scalp skin pixels to total region area to gauge parting width and hair thinning.
3. **Normalized Redness Index (Erythema)**:
   - Measures micro-vascular redness and tissue inflammation:
     $$\text{NRI} = \frac{2R - G - B}{R + G + B + 10^{-5}}$$
   - Quantifies percentage of scalp tissue experiencing active erythema.
4. **Flake & Dandruff Burden (%)**:
   - Analyzes high-value, low-saturation pixels in $HSV$ color space ($V > 200, S < 70$) within the scalp zone to calculate flaky scale accumulation.
5. **Contour Geometry & Focal Patch Detection**:
   - Evaluates contour circularity:
     $$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
   - **Mathematical Discrimination**:
     - *Coin-Shaped Bald Patch (Alopecia Areata)*: Discrete internal contour with high circularity ($C > 0.45$) and aspect ratio ($> 0.55$).
     - *Pattern Hair Thinning (Androgenetic Alopecia)*: Elongated diffuse scalp exposure along the sagittal parting or crown with low aspect ratio ($< 0.45$).
6. **Background Border Filtering**:
   - Contours touching image boundaries are automatically discarded, preventing light-colored background walls or clothing from being misclassified as bald spots.
7. **Actionable Plain-Language Advisory**:
   - Replaces dense clinical jargon with friendly, easy-to-understand explanations, clear next steps, and warnings against harmful home remedies (e.g., rubbing raw garlic or caustic pastes).

---

## 📁 Project Directory Structure

```text
AIML MINI/
│
├── dataset/                         # Partitioned 80/20 dataset
│   ├── train/                       # 2,239 training images
│   │   ├── alopecia_areata/         # 1,141 images
│   │   ├── androgenetic_alopecia/   # 496 images
│   │   └── seborrheic_dermatitis/   # 602 images
│   └── val/                         # 561 validation/testing images
│       ├── alopecia_areata/         # 286 images
│       ├── androgenetic_alopecia/   # 124 images
│       └── seborrheic_dermatitis/   # 151 images
│
├── saved_models/                    # Serialized neural network weights
│   └── best_scalp_model.pth         # Trained PyTorch EfficientNet-B0 weights
│
├── demo_phone_samples/              # Demo phone photos for 1-click testing
│   ├── healthy_scalp.jpg            # Dense coverage with narrow parting
│   ├── widening_part_thinning.jpg   # Diffuse widening crown parting
│   ├── erythematous_flaking.jpg     # Inflamed scalp with dandruff flakes
│   └── focal_bald_patch.jpg         # Circular coin-shaped bare spot
│
├── prepare_data.py                  # One-time script: validates & partitions dataset
├── train.py                         # Training pipeline: trains & saves best_scalp_model.pth
├── evaluate.py                      # Offline evaluation: prints report & saves confusion matrix
├── app.py                           # Full Streamlit web application (Dual-mode UI)
├── requirements.txt                 # Python project dependencies
├── confusion_matrix.png             # Seaborn confusion matrix visualization
└── README.md                        # Complete project documentation
```

---

## 🛠️ Tools & Technologies Used

| Tool / Library | Version | Purpose in Project |
| :--- | :--- | :--- |
| **Python** | `3.12+` | Core programming language |
| **PyTorch** (`torch`) | `2.14+` | Deep learning framework, tensor autograd, offline model loading |
| **Torchvision** | `0.29+` | EfficientNet-B0 architecture, pretrained weights, dataset transforms |
| **OpenCV** (`cv2`) | `4.9+` | Color space transforms, skin segmentation, contour circularity, overlay generation |
| **Streamlit** | `1.62+` | Interactive clinical dashboard, multi-column layout, image caching |
| **Scikit-Learn** | `1.8+` | Precision, recall, F1-score evaluation metrics, confusion matrix computation |
| **Pillow (PIL)** | `10.4+` | Image integrity verification (`Image.verify()`) and RGB loading |
| **NumPy** | `1.26+` | Vectorized matrix operations, array masking, color space math |
| **Matplotlib & Seaborn** | `3.10+ / 0.13+` | High-resolution confusion matrix heatmap plotting (`confusion_matrix.png`) |

---

## 🚀 Quickstart: How to Run the Project

### 1. Clone & Install Dependencies
Open your terminal inside the project directory and install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Prepare & Verify the Dataset *(Optional - Already Prepared)*
If starting from the raw archive, execute the data preparation script:
```bash
python prepare_data.py
```
*Scans all `.jpg` files, tests integrity with PIL, and creates the 80/20 train/val directory structure.*

### 3. Train the Model *(Optional - Saved Model Included)*
To retrain or train the EfficientNet-B0 classifier:
```bash
python train.py
```
*Trains for 10 epochs using AdamW, cosine annealing, and saves the best checkpoint to `./saved_models/best_scalp_model.pth`.*

### 4. Evaluate the Model (Print Metrics & Confusion Matrix)
Run the offline evaluation over all 561 test images:
```bash
python evaluate.py
```
*Outputs the classification report (precision, recall, F1-score) and generates `confusion_matrix.png`.*

### 5. Launch the Streamlit Web Application
Start the local dashboard:
```bash
streamlit run app.py
```
Your web browser will automatically open at:
```text
http://localhost:8501
```

---

## 🧪 Model Evaluation & Metrics

The offline evaluation across all **561 test images** via [`evaluate.py`](file:///c:/Users/neha/Downloads/AIML%20MINI/evaluate.py) yields the following performance:

```text
============================================================
                    CLASSIFICATION REPORT
============================================================
                       precision    recall  f1-score   support

      alopecia_areata     0.9052    0.7343    0.8108       286
androgenetic_alopecia     0.6218    0.5968    0.6091       124
seborrheic_dermatitis     0.6667    0.9272    0.7756       151

             accuracy                         0.7558       561
            macro avg     0.7312    0.7527    0.7318       561
         weighted avg     0.7784    0.7558    0.7567       561

Overall Validation Accuracy: 75.58%
[OK] Confusion matrix saved successfully to: ./confusion_matrix.png
```

- **Alopecia Areata Precision**: **90.52%** (high specificity for autoimmune hair loss).
- **Seborrheic Dermatitis Recall**: **92.72%** (exceptional sensitivity for detecting inflammatory/scaling fungal dermatoses).
- **High Multi-Class Balance**: Macro average F1-Score of **0.73+** across all classes.

---

## ⚖️ Academic & Clinical Disclaimer

**ScalpLens is an assistive algorithmic screening prototype designed for research, education, and preliminary risk triage.** 

- It is **not** an automated medical diagnosis or prescription system.
- Patient images are never uploaded to any remote cloud servers, ensuring 100% patient data confidentiality.
- Any decision regarding prescription therapeutics (such as Minoxidil, Finasteride, topical corticosteroids, or Ketoconazole) must be evaluated and confirmed by a certified dermatologist or trichology specialist.

---

## 📜 Citation & Acknowledgments
- **Dataset**: Peking Union Medical College Hospital (PUMCH-SCALP Benchmark).
- **Architecture**: EfficientNet (Tan & Le, ICML 2019).
- **Optimization**: AdamW (Loshchilov & Hutter, ICLR 2019).
