# WaveMesh-Diff Pipeline Guide

Complete guide for running the WaveMesh-Diff pipeline in Google Colab or Local environment.

---

## 🚀 Quick Start

### Google Colab (One Cell Setup)

```python
# Install dependencies
!pip install -q PyWavelets trimesh scikit-image scipy numpy torch torchvision rtree pillow pyyaml

# Clone repository
!git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf

# Run complete pipeline
!python run_pipeline.py

# Or run specific tests
!python run_pipeline.py --resolution 128 --test-diffusion
```

### Local Machine

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py

# Or with custom options
python run_pipeline.py --mesh path/to/mesh.obj --resolution 256 --test-diffusion
```

---

## 📊 Pipeline Options

### Basic Usage

```bash
# Run all modules with defaults
python run_pipeline.py

# Run with custom mesh
python run_pipeline.py --mesh bunny.obj

# Run with higher resolution
python run_pipeline.py --resolution 128

# Save outputs to custom directory
python run_pipeline.py --output ./results
```

### Module A Options (Wavelet Transform)

```bash
# Custom resolution and threshold
python run_pipeline.py --resolution 256 --threshold 0.0005

# Skip Module A (only test neural networks)
python run_pipeline.py --skip-module-a
```

### Modules B & C Options (Neural Networks)

```bash
# Custom model size
python run_pipeline.py \
  --encoder-channels 32 64 128 256 \
  --decoder-channels 256 128 64 32

# Enable attention mechanism
python run_pipeline.py --use-attention

# Test diffusion sampling
python run_pipeline.py --test-diffusion --diffusion-steps 200

# Custom spatial size and points
python run_pipeline.py --spatial-size 32 --num-points 500

# Skip Modules B & C (only test wavelets)
python run_pipeline.py --skip-module-bc
```

### Complete Example

```bash
# Full pipeline with all options
python run_pipeline.py \
  --mesh bunny.obj \
  --resolution 128 \
  --threshold 0.001 \
  --encoder-channels 16 32 64 128 \
  --decoder-channels 128 64 32 16 \
  --use-attention \
  --test-diffusion \
  --diffusion-steps 100 \
  --spatial-size 32 \
  --num-points 200 \
  --output ./results \
  --verbose
```

---

## 📈 What the Pipeline Does

### Module A: Wavelet Transform Pipeline

1. **Load/Create Mesh** - Input 3D mesh or create test sphere
2. **Mesh → SDF** - Convert to signed distance field
3. **SDF → Sparse Wavelet** - Apply 3D wavelet transform with thresholding
4. **Sparse Wavelet → SDF** - Reconstruct from sparse representation
5. **SDF → Mesh** - Generate final mesh
6. **Evaluate** - Calculate compression ratio and reconstruction error

**Outputs:**

- `output/01_original.obj` - Original mesh
- `output/02_reconstructed.obj` - Reconstructed mesh
- `output/statistics.txt` - Detailed statistics

### Modules B & C: Neural Network Pipeline

1. **Check Backend** - Detect spconv or dense fallback
2. **Create Model** - Build WaveMeshUNet with specified architecture
3. **Generate Test Data** - Create sparse 3D tensor data
4. **Forward Pass** - Test neural network denoising
5. **Diffusion Test** (optional) - Test diffusion model sampling

**Outputs:**

- Model architecture summary
- Backend information (spconv or dense fallback)
- Forward pass results
- Diffusion model status

---

## 📊 Expected Results

### Module A Output

```
MODULE A: Wavelet Transform Pipeline
====================================================================

📁 Loading mesh from: bunny.obj
✅ Mesh loaded: 2503 vertices, 4968 faces

📊 Converting mesh to SDF (resolution: 128³)...
✅ SDF grid created: (128, 128, 128)
   SDF range: [-0.433, 0.433]

🌊 Applying wavelet transform (threshold: 0.001)...
✅ Wavelet transform complete:
   Total coefficients: 2,097,152
   Non-zero coefficients: 21,043
   Sparsity: 99.00%
   Compression ratio: 99.6x

🔄 Reconstructing SDF from sparse wavelet...
✅ Reconstruction complete:
   MSE: 0.000089
   Max error: 0.012

💾 Saving outputs to: output
   ✅ Saved original mesh: output/01_original.obj
   ✅ Saved reconstructed mesh: output/02_reconstructed.obj
   ✅ Saved statistics: output/statistics.txt
```

### Modules B & C Output

```
MODULES B & C: Neural Network Pipeline
====================================================================

🔧 Backend: dense_fallback
📊 Performance: suboptimal (10-100x slower)
⚠️  Note: Using dense fallback (slower but functional)
   For production, install spconv with GPU support

🧠 Creating WaveMeshUNet model...
✅ Model created with 1,234,567 parameters

📊 Creating test sparse data...
✅ Test data created:
   Batch size: 1
   Num points: 100
   Spatial shape: (16, 16, 16)
   Input features: torch.Size([100, 1])

🔄 Running forward pass...
✅ Forward pass successful!
   Input: torch.Size([100, 1])
   Output: torch.Size([100, 1])

🌀 Creating Gaussian Diffusion model...
✅ Diffusion model created with 100 timesteps
```

---

## 🎯 Common Use Cases

### 1. Quick Test (Recommended First Run)

```bash
# Fast test with small resolution
python run_pipeline.py --resolution 64
```

**Time:** ~30 seconds  
**Purpose:** Verify everything works

### 2. High Quality Wavelet Compression

```bash
# High resolution with fine threshold
python run_pipeline.py --resolution 256 --threshold 0.0005
```

**Time:** ~5 minutes  
**Purpose:** Best quality reconstruction

### 3. Test Neural Networks Only

```bash
# Skip wavelet, test U-Net and Diffusion
python run_pipeline.py --skip-module-a --test-diffusion
```

**Time:** ~10 seconds  
**Purpose:** Test neural network components

### 4. Complete Pipeline with Your Mesh

```bash
# Process custom mesh through full pipeline
python run_pipeline.py \
  --mesh your_model.obj \
  --resolution 128 \
  --test-diffusion \
  --output ./my_results
```

**Time:** ~2 minutes  
**Purpose:** Full pipeline on custom data

### 5. Large Model Test

```bash
# Test with larger neural network
python run_pipeline.py \
  --encoder-channels 32 64 128 256 512 \
  --decoder-channels 512 256 128 64 32 \
  --use-attention \
  --spatial-size 32 \
  --num-points 500
```

**Time:** ~1 minute  
**Purpose:** Test scaling to larger models

---

## 🐛 Troubleshooting

### Issue: "Module not found"

**Solution:**

```python
import sys
sys.path.insert(0, '.')
```

### Issue: "Out of memory" in Colab

**Solution:** Use smaller resolution or model

```bash
python run_pipeline.py --resolution 32 --encoder-channels 8 16 32
```

### Issue: Dense fallback is slow

**Expected!** This is normal in Colab. For production:

```bash
# Install spconv on local machine with GPU
pip install spconv-cu118  # or spconv-cu121
```

### Issue: Mesh loading fails

**Solution:** Make sure mesh is in supported format (.obj, .stl, .ply)

```bash
# Convert mesh if needed
trimesh path/to/mesh.fbx --export mesh.obj
```

---

## 📚 Understanding the Output

### Sparsity

Percentage of wavelet coefficients that are zero (removed).

- **95-98%**: Normal for simple shapes
- **98-99%**: Good for complex shapes
- **99%+**: Excellent for high-resolution grids

### Compression Ratio

How much smaller the sparse representation is.

- **50-100x**: Typical for 64³ resolution
- **100-500x**: Good for 128³ resolution
- **500x+**: Excellent for 256³+ resolution

### MSE (Mean Squared Error)

Average squared difference between original and reconstructed SDF.

- **< 0.001**: Excellent quality
- **< 0.01**: Good quality
- **> 0.1**: May have visible artifacts

### Backend

- **spconv**: Optimal performance (local GPU)
- **dense_fallback**: Slower but functional (Colab)

---

## 🎓 Next Steps

After running the pipeline:

1. **Visualize Results**

   - Download `output/*.obj` files
   - Open in MeshLab, Blender, or any 3D viewer
   - Compare original vs reconstructed

2. **Experiment**

   - Try different thresholds (0.0001 to 0.01)
   - Try different resolutions (32, 64, 128, 256)
   - Test with your own meshes

3. **Understand Trade-offs**

   - Higher resolution = Better quality, more memory
   - Lower threshold = Less compression, better quality
   - Larger models = More capacity, slower training

4. **Production Use**
   - Install spconv for optimal performance
   - Use GPU for faster processing
   - Train diffusion model on dataset

---

## 📞 Help & Documentation

- **Setup Guide:** [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **All Docs:** [DOCS_INDEX.md](DOCS_INDEX.md)

---

**Happy experimenting! 🚀**
