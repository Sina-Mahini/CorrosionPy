# CorrosionPy 🔬

CorrosionPy is an AI-based tool for characterizing and subtyping corrosion failures in engineering materials.

## 📌 Overview

CorrosionPy is a deep learning-based tool that supports researchers and engineers in the characterization and automated classification of fundamental corrosion types using scanning electron microscope (SEM) images. The current demo version uses a Convolutional Neural Network (CNN) trained on a hybrid dataset composed of real SEM micrographs and scientifically realistic augmented images.

## ✨ Features

- **Automated Corrosion Classification** - Characterize and subtype corrosion failure patterns
- **Trained on Expert-Labeled Data** - Model built on verified corrosion SEM images
- **Augmented Training Set** - Robust performance even with limited experimental data
- **Ready-to-Use Demo** - Test hypotheses and validate against classified benchmark data
- **Extensible Framework** - Easily retrain or fine-tune on your own datasets

## 🎯 Use Cases

- Failure analysis in materials science and engineering
- Corrosion research and characterization
- Educational tool for electron microscopy image interpretation
- Benchmark dataset for corrosion classification algorithms

## 🚀 Quick Start

```python
# Example coming soon
from corrosionpy import CorrosionClassifier

model = CorrosionClassifier.load_pretrained()
result = model.predict("sem_image.tif")
print(f"Failure type: {result.class}", f"Confidence: {result.confidence:.2f}")
