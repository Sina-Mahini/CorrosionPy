# CorrosionPy 🔬

CorrosionPy is a machine learning framework developed for the automated morphological classification of corrosion failure modes in engineering materials. 
By leveraging a custom Deep Convolutional Neural Network (DCNN) architecture, the package enables high-fidelity characterization of degradation patterns directly from Scanning Electron Microscopy (SEM) imagery.
This demo employs a CNN trained on a mixed dataset of real micrographs and scientifically validated augmented images.
The tool is designed to bridge the gap between raw microscopic data and metallurgical failure analysis.


## ✨ Features

- **Automated Classification** - CNN-based characterization of corrosion patterns
- **Expert-Labeled Training** - Model validated on verified SEM micrographs
- **Augmented Dataset** - Extending the experimental data
- **Extensible Framework** - Built on PyTorch, allowing for easy fine-tuning or transfer learning on proprietary alloy-specific datasets.


## 🚀 Quick Start
The pipeline requires the following dependencies:
* `torch` (PyTorch)
* `torchvision`
* `matplotlib`
* `numpy`

## How to use
```python
from corrosionpy import CorrosionClassifier

model = CorrosionClassifier.load_pretrained()
result = model.predict("sem_image.tif")
print(f"Failure type: {result.class} | Confidence: {result.confidence:.2f}")
```

## Alternatively
```bash
git clone [https://github.com/yourusername/CorrosionPy.git](https://github.com/yourusername/CorrosionPy.git)
cd CorrosionPy
pip install torch torchvision matplotlib
