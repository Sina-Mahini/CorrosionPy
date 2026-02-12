# CorrosionPy 🔬

CorrosionPy is an AI-based tool for subtyping corrosive failures and characterizing failure types in materials.

## 📌 Overview

CorrosionPy is a deep learning-based tool designed to assist researchers and engineers in the automated classification of corrosion failure types from SEM images. The current demo version employs a Convolutional Neural Network (CNN) trained on a hybrid dataset of real and synthetically augmented corrosion microscopy images.

## ✨ Features

- **Automated Corrosion Classification** - Identify and subtype corrosive failure patterns
- **Trained on Expert-Labeled Data** - Model built on verified corrosion SEM images
- **Augmented Training Set** - Robust performance even with limited experimental data
- **Ready-to-Use Demo** - Test hypotheses and validate against classified benchmark data
- **Extensible Framework** - Easily retrain or fine-tune on your own datasets

## 🎯 Use Cases

- Failure analysis in materials science
- Corrosion research and characterization
- Educational tool for microscopy image interpretation
- Benchmark dataset for corrosion classification algorithms

## 🚀 Quick Start

```python
# Example coming soon
from corrosionpy import CorrosionClassifier

model = CorrosionClassifier.load_pretrained()
result = model.predict("sem_image.tif")
print(f"Failure type: {result.class}", f"Confidence: {result.confidence:.2f}")
