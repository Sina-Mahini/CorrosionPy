# CorrosionPy 🔬

CorrosionPy is an AI-based tool for characterizing and subtyping corrosion failures in engineering materials.

## 📌 Overview

Deep learning-based tool for automated classification of fundamental corrosion types from SEM images. Demo version employs a CNN trained on a hybrid dataset of real micrographs and scientifically validated augmented images.

## ✨ Features

- **Automated Classification** - CNN-based characterization of corrosion patterns
- **Expert-Labeled Training** - Model validated on verified SEM micrographs
- **Augmented Dataset** - Robust performance with limited experimental data
- **Extensible Framework** - Fine-tune on custom datasets

## 🎯 Use Cases

- Failure analysis in materials science
- Corrosion research
- SEM image interpretation

## 🚀 Quick Start

```python
from corrosionpy import CorrosionClassifier

model = CorrosionClassifier.load_pretrained()
result = model.predict("sem_image.tif")
print(f"Failure type: {result.class} | Confidence: {result.confidence:.2f}")
