#  COVID-19 Chest X-Ray Classifier

A Convolutional Neural Network (CNN) built from scratch to classify chest X-ray images into 4 categories: **COVID-19**, **Normal**, **Lung Opacity**, and **Viral Pneumonia** — achieving **88% accuracy**.

---

##  Overview

This project uses deep learning to assist in the automated diagnosis of lung conditions from chest radiography images. Given the critical need for rapid COVID-19 screening, this model provides a fast and reliable classification pipeline trained on thousands of X-ray images.

---

##  Dataset

**COVID-19 Radiography Database** — available on [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

| Class | Description |
|---|---|
| `Normal` | Healthy chest X-rays |
| `COVID` | COVID-19 positive cases |
| `Lung_Opacity` | Non-COVID lung opacity |
| `Viral Pneumonia` | Viral pneumonia cases |

Images are PNG format at 1024×1024 pixels, resized to **100×100** for training.

---

## Model Architecture

A custom CNN built with TensorFlow/Keras:
```
Input (100x100x1 grayscale)
│
├── Conv2D(32, 3x3, ReLU) → MaxPooling(2x2)
├── Conv2D(64, 3x3, ReLU) → MaxPooling(2x2)
├── Conv2D(128, 3x3, ReLU) → MaxPooling(2x2)
├── Conv2D(64, 3x3, ReLU) → MaxPooling(2x2)
│
├── Flatten
├── Dense(128, ReLU)
├── Dropout(0.5)
└── Dense(4, Softmax)
```

**Optimizer:** Adam | **Loss:** Sparse Categorical Crossentropy | **Batch Size:** 32 | **Max Epochs:** 10

---

## Results

**Overall Accuracy: 88%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.89 | 0.92 | 0.90 | 1025 |
| COVID | 0.86 | 0.88 | 0.87 | 374 |
| Lung Opacity | 0.86 | 0.80 | 0.83 | 582 |
| Viral Pneumonia | 0.93 | 0.93 | 0.93 | 136 |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | **2117** |

---

##  Data Pipeline

1. **Load** — Images read with OpenCV, converted to grayscale
2. **Resize** — Standardized to 100×100 pixels
3. **Normalize** — Pixel values scaled to [0, 1]
4. **Split:**
   - 80% → Training
   - 10% → Validation
   - 10% → Test

---

##  Requirements
```bash
pip install kagglehub opencv-python numpy matplotlib scikit-learn tensorflow
```

| Library | Purpose |
|---|---|
| `kagglehub` | Dataset download |
| `opencv-python` | Image loading & preprocessing |
| `numpy` | Array operations |
| `matplotlib` | Visualization |
| `scikit-learn` | Train/test split & metrics |
| `tensorflow` / `keras` | CNN model building & training |

---

##  How to Run

**1. Download the dataset:**
```python
import kagglehub
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
```

**2.** Run the notebook on Kaggle or locally (update `data_path` if running locally).

**3.** The model will train and output accuracy/loss curves + a full classification report.

---

##  Notes

- Model trained on **grayscale** images (single channel)
- **Early Stopping** (patience=3) monitors `val_loss` to prevent overfitting
- **Dropout(0.5)** added for regularization

---

## License

Dataset is publicly available on Kaggle for research purposes. Model code is open-source and free to use.