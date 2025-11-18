# WaveMesh-Diff Setup Guide

Complete guide for running WaveMesh-Diff in **Google Colab** or on your **Local Machine**.

---

## 🎯 Choose Your Environment

- **[Google Colab Setup](#google-colab-setup)** - No installation needed, run in browser
- **[Local Machine Setup](#local-machine-setup)** - Full performance with GPU

---

# Google Colab Setup

## ✅ What Works in Colab

All three modules work in Google Colab without spconv compilation!

| Module                          | Status        | Performance    | Notes               |
| ------------------------------- | ------------- | -------------- | ------------------- |
| **Module A: Wavelet Transform** | ✅ Full Speed | 100%           | Optimal performance |
| **Module B: Sparse U-Net**      | ✅ Functional | ~10-50x slower | Dense fallback mode |
| **Module C: Diffusion Model**   | ✅ Functional | ~10-50x slower | Dense fallback mode |

**Dense Fallback Mode**: Modules B & C automatically use PyTorch dense operations when spconv isn't available. Perfect for learning and testing!

**Recommendation:**

- ✅ **Use Colab for**: Learning, testing, prototyping
- ⚡ **Use Local GPU for**: Training, production, large models

---

## 🚀 Quick Start (Colab)

### Option 1: One-Cell Setup (Public Repository)

Copy this into a Colab cell:

```python
# Install dependencies
!pip install -q PyWavelets trimesh scikit-image scipy numpy torch torchvision rtree pillow pyyaml

# Configure for headless environment
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Clone repository
!git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf

# Test Module A (Wavelet Transform) - Full speed!
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64

print("✅ Module A test complete!")
```

### Option 2: Test All Modules (Colab)

```python
# After running setup above, test Modules B & C
import torch
from models import WaveMeshUNet, GaussianDiffusion
from models.spconv_compat import get_backend_info, sparse_tensor

# Check which backend is active
info = get_backend_info()
print(f"✅ Using: {info['backend']}")
print(f"📊 Performance: {info['performance']}")

# Create a small U-Net model
model = WaveMeshUNet(
    in_channels=1,
    out_channels=1,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16],
    use_attention=False
)
print(f"✅ Created WaveMeshUNet with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
batch_size = 1
num_points = 50
features = torch.randn(num_points, 1)
indices = torch.cat([
    torch.zeros(num_points, 1, dtype=torch.long),
    torch.randint(0, 16, (num_points, 3))
], dim=1)

sp_input = sparse_tensor(features, indices, (16, 16, 16), batch_size)
timesteps = torch.zeros(batch_size, dtype=torch.long)

with torch.no_grad():
    output = model(sp_input, timesteps)

print(f"✅ Forward pass successful!")
print(f"   Input: {sp_input.features.shape}, Output: {output.features.shape}")
```

---

## 🔐 Private Repository Setup (Colab)

If your repository is private, you need authentication:

### Step 1: Create GitHub Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "Colab Access"
4. Check the `repo` scope
5. Generate and **copy the token**

### Step 2: Add Token to Colab

```python
# In Colab: Click 🔑 (Secrets) in left sidebar
# Add secret: Name = "GH_TOKEN", Value = <your token>
# Enable notebook access

from google.colab import userdata
GH_TOKEN = userdata.get('GH_TOKEN')

# Clone with authentication
!git clone https://{GH_TOKEN}@github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf
```

---

## 📊 Expected Results (Colab)

### Module A Test Output

```
Mesh statistics:
  Vertices: 642
  Faces: 1280
  Bounds: [-1.0, 1.0]

SDF statistics:
  Shape: (64, 64, 64)
  Method: simple (Colab compatible)

Wavelet statistics:
  Coefficients: 262,144 total
  Non-zero: 5,234 (2.0%)
  Sparsity: 98.0%
  Compression ratio: 50.1x

Reconstruction quality:
  MSE: 0.000123
  Max error: 0.045
  ✅ Reconstruction successful!
```

### Module B/C Test Output

```
✅ Using: dense_fallback
📊 Performance: suboptimal (10-100x slower)
✅ Created WaveMeshUNet with 1,234,567 parameters
✅ Forward pass successful!
   Input: torch.Size([50, 1]), Output: torch.Size([50, 1])
```

---

## 🐛 Troubleshooting (Colab)

### "rtree not found"

```python
!pip install rtree
```

### "Out of memory"

Use smaller models:

```python
encoder_channels = [8, 16, 32]  # Instead of [32, 64, 128, 256]
```

### "Module not found"

Make sure you're in the repo directory:

```python
%cd WaveMeshDf
import sys
sys.path.insert(0, '.')
```

### Want faster performance?

Modules B & C are slower in Colab due to dense fallback. For production:

- Use local GPU with spconv installed (see below)

---

# Local Machine Setup

## 💻 Full Performance Setup

For production use, training, and optimal performance.

---

## Prerequisites

- Python 3.8+
- CUDA 11.8 or 12.1 (for GPU acceleration)
- Git

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
cd WaveMeshDf
```

### Step 2: Install PyTorch

**For CUDA 11.8:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**

```bash
pip install torch torchvision torchaudio
```

### Step 3: Install spconv (For Optimal Performance)

**For CUDA 11.8:**

```bash
pip install spconv-cu118
```

**For CUDA 12.1:**

```bash
pip install spconv-cu121
```

**Note**: If spconv installation fails, the system will automatically use dense fallback mode (slower but functional).

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- PyWavelets (wavelet transforms)
- trimesh (mesh processing)
- scikit-image (marching cubes)
- scipy, numpy (numerical operations)
- pillow (image processing)
- pyyaml (configuration)

---

## ✅ Verify Installation

```bash
# Test Module A
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128

# Test Modules B & C
python tests/test_modules_bc.py
```

**Expected Output:**

```
✅ All tests passed!
Module A: Wavelet Transform ✅
Module B: Sparse U-Net ✅
Module C: Diffusion Model ✅

Backend: spconv (optimal performance)
```

---

## 📖 Usage Examples

### Test 1: Simple Sphere (Recommended First Test)

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128
```

Creates test sphere, runs full pipeline, outputs 3 meshes to `output/`:

- `01_original.obj` - Input mesh
- `02_from_sdf.obj` - Baseline (no compression)
- `03_reconstructed.obj` - From sparse wavelet

### Test 2: Your Own Mesh

```bash
python tests/test_wavelet_pipeline.py --mesh path/to/mesh.obj --resolution 256
```

### Test 3: Find Optimal Threshold

```bash
python tests/test_wavelet_pipeline.py --mesh path/to/mesh.obj --test-thresholds
```

Tests thresholds from 0.0001 to 0.05 and shows compression vs quality trade-off.

### Test 4: High Resolution

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 512
```

**Note**: Higher resolution = better quality but slower and more memory.

---

## 🎯 Quick Tests

### Test Module A Only (Wavelet)

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64
```

Fast test (~10 seconds) to verify basic functionality.

### Test All Modules

```bash
# Test wavelet transform
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128

# Test neural networks
python tests/test_modules_bc.py

# Test compatibility layer
python tests/test_spconv_compat.py
```

---

## 📊 Expected Performance

### Module A: Wavelet Transform

| Resolution | Time  | Memory | Sparsity |
| ---------- | ----- | ------ | -------- |
| 64³        | ~5s   | ~100MB | 95-98%   |
| 128³       | ~15s  | ~500MB | 98-99%   |
| 256³       | ~60s  | ~2GB   | 99%+     |
| 512³       | ~300s | ~8GB   | 99.5%+   |

### Modules B & C: Neural Networks

**With spconv (local GPU):**

- Training: ~100 meshes/hour
- Inference: ~0.5s per mesh

**Without spconv (dense fallback):**

- Training: ~2-10 meshes/hour
- Inference: ~5-25s per mesh

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Wavelet settings
wavelet:
  type: "db4" # Wavelet family
  threshold: 0.001 # Sparsification threshold

# Model settings
model:
  encoder_channels: [32, 64, 128, 256]
  decoder_channels: [256, 128, 64, 32]
  use_attention: true

# Training settings
training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 100
```

---

## 🐛 Troubleshooting (Local)

### spconv installation fails

**Solution 1**: Use dense fallback (automatic)

- System detects missing spconv and uses PyTorch operations
- Slower but functional

**Solution 2**: Install from source

```bash
git clone https://github.com/traveller59/spconv.git
cd spconv
python setup.py bdist_wheel
pip install dist/*.whl
```

### CUDA out of memory

Reduce batch size or model size:

```python
encoder_channels = [16, 32, 64]  # Smaller model
batch_size = 1  # Process one at a time
```

### Import errors

Make sure you're in the repository root:

```bash
cd WaveMeshDf
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Mesh loading fails

Install additional dependencies:

```bash
pip install pyglet pyembree
```

---

## 📚 Next Steps

### Learn the Pipeline

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the design
2. Read [MODULE_A_SUMMARY.md](MODULE_A_SUMMARY.md) - Deep dive into wavelets
3. Experiment with different meshes and settings

### Development

1. Check [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current state
2. See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
3. Review [COLAB_COMPATIBILITY.md](COLAB_COMPATIBILITY.md) - Fallback implementation

### Production Use

1. Install spconv for optimal performance
2. Use GPU with CUDA support
3. Configure batch processing
4. Set up model checkpointing

---

## 📊 Performance Comparison

### Google Colab vs Local GPU

| Feature              | Google Colab      | Local GPU (with spconv) |
| -------------------- | ----------------- | ----------------------- |
| **Setup Time**       | Instant           | 15-30 min               |
| **Cost**             | Free              | Hardware cost           |
| **Module A Speed**   | ⚡ Fast           | ⚡ Fast                 |
| **Module B/C Speed** | 🐢 10-50x slower  | ⚡ Optimal              |
| **Memory Limit**     | ~12GB             | Your GPU                |
| **Session Time**     | 12 hours max      | Unlimited               |
| **Best For**         | Learning, testing | Training, production    |

---

## ❓ FAQ

### Q: Should I use Colab or local setup?

**Use Colab if:**

- You're learning or experimenting
- You don't have a GPU
- You want instant setup

**Use Local if:**

- You're training models
- You need full performance
- You have production workloads

### Q: Can I switch between Colab and local?

Yes! Your code works the same in both environments. The system automatically detects spconv availability.

### Q: Why is Colab slower for Modules B & C?

Colab can't compile spconv (missing dependencies), so we use dense PyTorch operations instead. This processes the entire 3D grid rather than just sparse points.

### Q: Is the dense fallback accurate?

Yes! It produces identical results to spconv, just slower. Perfect for testing and validation.

### Q: Can I use Module A at full speed in Colab?

Yes! Module A uses standard operations and runs at full speed in Colab.

---

## 🎉 Success Checklist

After setup, you should be able to:

- ✅ Run Module A tests (wavelet transform)
- ✅ Process your own meshes
- ✅ Import and use WaveMeshUNet
- ✅ Import and use GaussianDiffusion
- ✅ Run complete forward passes
- ✅ See spconv or dense_fallback backend info

---

## 📞 Getting Help

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review test files in `tests/` for usage examples
- Read documentation in `docs/` folder
- Check GitHub issues for known problems

---

**Happy experimenting! 🚀**

_Last updated: 2025-01-18_
