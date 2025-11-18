# WaveMesh-Diff - Complete Installation Guide

## 🚀 Quick Installation

### Minimum Requirements (Core Features)

```bash
pip install torch numpy pywt trimesh matplotlib scipy scikit-image
```

### Recommended Installation (Full Features)

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: DINOv2 encoder (better quality)
pip install transformers huggingface_hub accelerate

# Optional: GPU sparse operations (faster training)
pip install spconv-cu118  # For CUDA 11.8
# or
pip install spconv-cu117  # For CUDA 11.7
```

---

## 📦 Package Details

### Core Dependencies (Required)

| Package        | Version | Purpose                 |
| -------------- | ------- | ----------------------- |
| `torch`        | ≥2.0.0  | Deep learning framework |
| `numpy`        | ≥1.20.0 | Numerical operations    |
| `PyWavelets`   | ≥1.4.0  | 3D wavelet transform    |
| `trimesh`      | ≥3.20.0 | Mesh processing         |
| `matplotlib`   | ≥3.5.0  | Visualization           |
| `scipy`        | ≥1.9.0  | Scientific computing    |
| `scikit-image` | ≥0.19.0 | Marching cubes          |

### Optional Dependencies (Recommended)

| Package           | Purpose          | Impact if Missing         |
| ----------------- | ---------------- | ------------------------- |
| `transformers`    | DINOv2 encoder   | Falls back to CNN encoder |
| `huggingface_hub` | Model downloads  | Manual download needed    |
| `accelerate`      | Faster inference | Slower model loading      |
| `spconv`          | GPU sparse ops   | Uses dense operations     |

---

## 🔧 Installation Methods

### Method 1: Automatic (Recommended)

```bash
# Clone repository
git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
cd WaveMeshDf

# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies
python install_optional.py

# Verify installation
python verify_complete_setup.py
```

### Method 2: Conda Environment

```bash
# Create environment
conda create -n wavemesh python=3.10
conda activate wavemesh

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
python install_optional.py
```

### Method 3: Google Colab

```python
# In Colab notebook
!git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf
!pip install -q PyWavelets trimesh matplotlib rtree scipy scikit-image transformers
```

---

## ✅ Verify Installation

### Quick Test

```bash
python test_all_modules.py
```

Expected output:

```
Results: 4/4 modules passed
  Module A ✅ PASS
  Module B ✅ PASS
  Module C ✅ PASS
  Module D ✅ PASS
```

### Complete Verification

```bash
python verify_complete_setup.py
```

This tests:

- ✅ All imports
- ✅ Module functionality
- ✅ Wavelet pipeline
- ✅ Model forward pass
- ✅ Training components
- ✅ Dataset loading

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'pywt'"

```bash
pip install PyWavelets
```

### Issue 2: "ModuleNotFoundError: No module named 'rtree'"

**Windows:**

```bash
conda install -c conda-forge rtree
```

**Linux/Mac:**

```bash
sudo apt-get install libspatialindex-dev  # Linux
brew install spatialindex  # Mac
pip install rtree
```

### Issue 3: "transformers not available"

This is **OK**! The system will use a fallback CNN encoder.

To install DINOv2 support:

```bash
pip install transformers huggingface_hub
```

### Issue 4: "CUDA out of memory"

Reduce model size or batch size:

```bash
python train.py --resolution 16 --batch_size 2
```

### Issue 5: Debugger warnings about frozen modules

**Solution 1: Use environment variable**

```bash
# Windows
set PYDEVD_DISABLE_FILE_VALIDATION=1
python your_script.py

# Linux/Mac
export PYDEVD_DISABLE_FILE_VALIDATION=1
python your_script.py
```

**Solution 2: Python flag**

```bash
python -Xfrozen_modules=off your_script.py
```

**Solution 3: Add to project**
Create `.pdbrc.py` in project root (already included).

### Issue 6: spconv installation fails

This is optional. Try different CUDA versions:

```bash
# CUDA 11.8
pip install spconv-cu118

# CUDA 11.7
pip install spconv-cu117

# CPU only (slow)
# Just skip spconv - code will use dense operations
```

---

## 🔍 Check Installed Versions

```python
import torch
import numpy as np
import pywt
import trimesh

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
print(f"PyWavelets: {pywt.__version__}")
print(f"Trimesh: {trimesh.__version__}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: Not installed (using fallback)")
```

---

## 📚 Platform-Specific Instructions

### Windows

```bash
# Use Anaconda for easier installation
conda create -n wavemesh python=3.10
conda activate wavemesh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge rtree
pip install PyWavelets trimesh matplotlib scipy scikit-image
pip install transformers huggingface_hub
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev libspatialindex-dev

# Create virtual environment
python3 -m venv wavemesh_env
source wavemesh_env/bin/activate

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python install_optional.py
```

### macOS

```bash
# Install Homebrew dependencies
brew install spatialindex

# Create virtual environment
python3 -m venv wavemesh_env
source wavemesh_env/bin/activate

# Install packages
pip install torch torchvision torchaudio
pip install -r requirements.txt
python install_optional.py
```

---

## 🚀 Next Steps

After successful installation:

1. **Test Installation**

   ```bash
   python test_all_modules.py
   ```

2. **Download Data**

   ```bash
   python scripts/download_data.py --dataset modelnet40
   ```

3. **Run Quick Demo**

   ```bash
   python train.py --data_root data/ModelNet40 --debug --max_samples 20
   ```

4. **Or Use Colab**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoangNguyennnnnnn/WaveMeshDf/blob/main/colab_quickstart.ipynb)

---

## 💡 Tips

- **For development**: Use `--debug --max_samples 20` for fast iteration
- **For production**: Install all optional dependencies for best performance
- **GPU training**: Make sure CUDA is properly installed
- **Memory issues**: Reduce `--resolution` or `--batch_size`
- **Missing transformers**: It's OK! Fallback encoder works fine for testing

---

**Need help?** Open an issue: https://github.com/HoangNguyennnnnnn/WaveMeshDf/issues
