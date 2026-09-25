# Satellite Image Classification using EfficientNetB0

A deep learning project for **satellite image classification** using **TensorFlow/Keras** and a pretrained **EfficientNetB0** model.

The model classifies satellite images into four categories:

- `cloudy`
- `desert`
- `green_area`
- `water`

The project uses transfer learning followed by fine-tuning to improve classification performance.

## 📌 Project Overview

This project builds an image classification model that can identify the type of landscape/environment shown in a satellite image.

The workflow includes:

1. Installing the required Python libraries.
2. Downloading the satellite image dataset from Kaggle.
3. Loading and preprocessing the images.
4. Applying data augmentation.
5. Using ImageNet-pretrained EfficientNetB0 as the feature extractor.
6. Training a custom classification head.
7. Fine-tuning the upper layers of EfficientNetB0.
8. Evaluating the trained model on a test set.
9. Generating ROC curves and a confusion matrix.
10. Testing the model on individual images.
11. Saving the trained model as a `.keras` file.

## 🧠 Model Architecture

The project uses **EfficientNetB0** with ImageNet pretrained weights.

### Transfer Learning

Initially, the EfficientNetB0 base model is frozen and only the newly added classification layers are trained.

The classification head contains:

- Global Average Pooling
- Dropout (`0.3`)
- Dense layer (`256` units, ReLU)
- Batch Normalization
- Dropout (`0.3`)
- Final Dense layer with Softmax activation

### Fine-Tuning

After training the classification head, the model is fine-tuned.

Approximately the final **20% of EfficientNetB0 layers** are made trainable, while the earlier layers remain frozen.

A smaller learning rate (`1e-5`) is used during fine-tuning.

## 📂 Dataset

The project uses the **Satellite Image Classification** dataset available on Kaggle:

https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

The dataset contains four classes:

| Class | Description |
|---|---|
| `cloudy` | Cloudy satellite scenes |
| `desert` | Desert landscapes |
| `green_area` | Vegetation/green areas |
| `water` | Water bodies such as lakes and seas |

The notebook uses an 80/20 training-validation split.

During the recorded run:

- Total images used for training/validation: **5,631**
- Training images: **4,505**
- Validation images: **1,126**
- Test images: **576**

The dataset itself is not included in this repository. Download it from Kaggle when running the notebook.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- TensorFlow Hub
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Kaggle API
- Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```
Satellite-Image-Detection/
│
├── SatelliteImageDetection.ipynb
├── satellite_efficientnetb0.keras
├── requirements.txt
└── README.md
```

The model is evaluated using:

- Test accuracy
- Classification report
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curves
- Micro-average ROC/AUC

## 📊 Results

The recorded notebook run achieved:

```
Test Accuracy: 98.61%
Final Training Accuracy: 98.22%
```

The test set contained **576 images**.

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| cloudy | 1.00 | 1.00 | 1.00 |
| desert | 0.99 | 1.00 | 1.00 |
| green_area | 0.97 | 0.99 | 0.98 |
| water | 0.99 | 0.96 | 0.97 |
| **Overall accuracy** | | | **0.99** |

These results describe the particular training run stored in the notebook and may differ if the model is retrained.

## 🔍 Example Predictions

The notebook includes example predictions for:

- A green-area image
- A desert image
- A water image

Example recorded predictions:

```
green_area → 98.77% confidence
desert     → 100.00% confidence
water      → 99.58% confidence
```

Confidence values are model outputs and should not be interpreted as guaranteed correctness.

## 📈 Evaluation Visualizations

The notebook generates several visualizations:

### Accuracy Curve

Shows training and validation accuracy across training epochs and indicates when fine-tuning begins.

### ROC Curves

Generates one-vs-rest ROC curves for the four classes and also calculates a micro-average ROC curve.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

### Classification Report

Reports precision, recall, F1-score, and support for each class.

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Image size | `224 × 224` |
| Batch size | `32` |
| Initial optimizer | Adam |
| Initial training epochs | Up to `20` |
| Fine-tuning epochs | Up to `15` |
| Fine-tuning learning rate | `1e-5` |
| Dropout | `0.3` |
| Dense layer | `256` units |
| Backbone | EfficientNetB0 |
| Pretrained weights | ImageNet |
| Number of classes | `4` |
| Train/validation split | `80% / 20%` |

Early stopping and learning-rate reduction are used to control training.

## 🚀 Future Improvements

Possible extensions include:

- Deploying the model as a web application.
- Creating a REST API using Flask or FastAPI.
- Building a Streamlit interface for image classification.
- Adding more satellite image classes.
- Training with additional datasets.
- Performing more extensive hyperparameter tuning.
- Adding model explainability using Grad-CAM.
- Converting the model to TensorFlow Lite for lightweight deployment.
- Adding automated model evaluation during training.
- Creating a dedicated inference script for production use.

## ⚠️ Limitations

- The model is trained for the four classes present in the selected dataset.
- Predictions outside these categories may be unreliable.
- High confidence does not necessarily mean that a prediction is correct.
- Performance can change when images come from a different source, sensor, geographic region, or imaging condition.
- The reported metrics correspond to the recorded notebook run and are not a guarantee of performance on new datasets.
