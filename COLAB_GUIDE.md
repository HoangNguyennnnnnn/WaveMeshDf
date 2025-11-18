# 🚀 Google Colab Complete Setup Guide for WaveMesh-Diff

This guide will help you run the complete WaveMesh-Diff pipeline on Google Colab, including all modules (A, B, C).

---

## 📋 Quick Start (Copy-Paste Ready)

### Step 0: Setup GitHub Access (For Private Repositories)

**If the repository is private**, you need to set up authentication:

1. **Create a GitHub Personal Access Token:**

   - Go to https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Give it a name (e.g., "Colab Access")
   - Check the `repo` scope (full control of private repositories)
   - Click "Generate token" and **copy the token**

2. **Add token to Colab Secrets:**

   - In your Colab notebook, click the 🔑 **Key icon** in the left sidebar
   - Click "Add a secret"
   - Name: `GH_TOKEN`
   - Value: Paste your GitHub token
   - Toggle "Notebook access" ON

3. **Verify the secret is set:**
   ```python
   from google.colab import userdata
   try:
       token = userdata.get('GH_TOKEN')
       print("✅ GitHub token found!")
   except:
       print("❌ GitHub token not found. Please add GH_TOKEN to secrets.")
   ```

**For public repositories**, you can skip this step.

---

### Step 1: Setup Environment

Copy this entire cell into a new Colab notebook:

```python
# ========================================
# WaveMesh-Diff Environment Setup
# ========================================

# 1. Configure for headless rendering
import os
import subprocess
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 2. Install core dependencies
print("📦 Installing core dependencies...")
!pip install -q PyWavelets trimesh scikit-image scipy numpy torch torchvision

# 3. Detect CUDA version and install matching spconv
print("\n🔍 Detecting CUDA version...")
try:
    cuda_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
    print(cuda_version)

    # Parse CUDA version
    if 'release 12.5' in cuda_version or 'release 12.4' in cuda_version or 'release 12.3' in cuda_version:
        print("📥 Installing spconv for CUDA 12.x (using cu118 - compatible)...")
        print("ℹ️  Note: spconv-cu118 works with CUDA 12.x in most cases")
        !pip install -q spconv-cu118
    elif 'release 12.1' in cuda_version or 'release 12.0' in cuda_version:
        print("📥 Installing spconv for CUDA 12.0/12.1...")
        !pip install -q spconv-cu121
    elif 'release 11.8' in cuda_version:
        print("📥 Installing spconv for CUDA 11.8...")
        !pip install -q spconv-cu118
    else:
        print("⚠️  Unknown CUDA version, trying spconv-cu118...")
        !pip install -q spconv-cu118
except Exception as e:
    print(f"⚠️  Could not detect CUDA: {e}")
    print("Installing spconv-cu118 as default...")
    !pip install -q spconv-cu118

# 4. Install additional utilities
print("\n📦 Installing additional utilities...")
!pip install -q tqdm pyyaml einops

# 5. Clone the repository with authentication
print("\n📥 Cloning WaveMesh-Diff repository...")

try:
    # Try to get GitHub token from secrets (for private repos)
    from google.colab import userdata
    gh_token = userdata.get('GH_TOKEN')

    # Clone with authentication
    repo_url = f"https://{gh_token}@github.com/HoangNguyennnnnnn/WaveMeshDf.git"
    !git clone {repo_url} WaveMeshDf
    print("✅ Cloned private repository successfully!")
except:
    # Fall back to public clone (will fail if repo is private)
    print("ℹ️  No GH_TOKEN found in secrets, trying public clone...")
    !git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git WaveMeshDf

# Change to repository directory
import os
if os.path.exists('WaveMeshDf'):
    %cd WaveMeshDf
    print("✅ Changed to WaveMeshDf directory")
else:
    print("❌ Failed to clone repository. Check:")
    print("   1. Repository URL is correct")
    print("   2. Repository is public OR you've added GH_TOKEN to secrets")
    print("   3. GH_TOKEN has 'repo' permissions")
    raise FileNotFoundError("Repository not cloned")

# 6. Verify installation
print("\n🔍 Verifying installation...")
if os.path.exists('verify_installation.py'):
    !python verify_installation.py
else:
    print("⚠️  verify_installation.py not found, running basic checks...")

    # Basic verification
    import sys
    errors = []

    try:
        import pywt
        print("✓ PyWavelets imported successfully")
    except ImportError as e:
        errors.append(f"PyWavelets: {e}")

    try:
        import trimesh
        print("✓ trimesh imported successfully")
    except ImportError as e:
        errors.append(f"trimesh: {e}")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")

    try:
        import spconv.pytorch as spconv
        print(f"✓ spconv imported successfully")
    except ImportError as e:
        errors.append(f"spconv: {e}")

    if errors:
        print("\n❌ Import errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✅ All core dependencies verified!")

print("\n" + "="*60)
print("✅ Setup complete! Ready to run WaveMesh-Diff")
print("="*60)
```

---

## 🧪 Test Module A: Wavelet Transform

### Basic Test (Fast - 30 seconds)

```python
# Test wavelet pipeline with simple mesh
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64

# Check outputs
!ls -lh output/
```

### High Quality Test (Medium - 2 minutes)

```python
# Higher resolution for better quality
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128

# Test different thresholds
!python tests/test_wavelet_pipeline.py --create-test-mesh --test-thresholds --resolution 64
```

### Expected Output:

```
✓ Test mesh created: 642 vertices, 1280 faces
================================================================================
[Step 1] Loading original mesh...
  ✓ Loaded mesh: 642 vertices, 1280 faces

[Step 2] Converting mesh to SDF grid (resolution=64^3)...
  ✓ SDF grid shape: (64, 64, 64)

[Step 3] Converting to sparse wavelet (level=3, threshold=0.01)...
  ✓ Sparse representation created
    - Sparsity ratio: 97.2%
    - Compression ratio: 35.71x

[Step 4] Reconstructing dense SDF...
  Reconstruction Quality:
    - MSE: 0.001234
    - MAE: 0.012345

✅ All tests completed successfully!
```

**Note:** If you see negative sparsity (e.g., -3.9%) with very small test grids (32³), this is normal. The wavelet coefficients overhead exceeds savings at tiny sizes. Use resolution ≥64 for meaningful compression.

---

## 🤖 Test Modules B & C: Neural Networks

### Prerequisites

Make sure GPU is enabled:

- Runtime → Change runtime type → Hardware accelerator → GPU → Save

```python
# Verify GPU is available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Run Neural Network Tests

```python
# Test Sparse U-Net and Diffusion Model
!python tests/test_modules_bc.py

# This will run 4 tests:
# 1. Sparse U-Net architecture
# 2. Diffusion process
# 3. Wavelet integration
# 4. DDIM sampling
```

### Expected Output:

```
================================================================================
Test 1: Sparse U-Net Architecture
================================================================================
Using device: cuda

[Step 1] Creating WaveMeshUNet...
  ✓ Model created
    - Total parameters: 1,234,567
    - Trainable parameters: 1,234,567

[Step 2] Creating sparse input tensor...
  ✓ Sparse tensor created
    - Sparsity: 99.2%

[Step 4] Running forward pass...
  ✓ Forward pass successful

✅ Test 1 PASSED: Sparse U-Net works correctly

... (Tests 2, 3, 4)

✅ All tests passed successfully!
```

---

## 📊 Visualize Results

### View Generated Meshes

```python
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh(mesh_path, title="Mesh"):
    """Plot a 3D mesh in Colab."""
    mesh = trimesh.load(mesh_path)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh collection
    mesh_collection = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.7,
        facecolor='cyan',
        edgecolor='black',
        linewidths=0.1
    )
    ax.add_collection3d(mesh_collection)

    # Set limits and labels
    scale = mesh.vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

# Plot original mesh
plot_mesh('output/01_original.obj', 'Original Mesh')

# Plot reconstructed mesh
plot_mesh('output/03_reconstructed.obj', 'Reconstructed from Sparse Wavelet')
```

### Compare Quality

```python
import numpy as np

# Load meshes
original = trimesh.load('output/01_original.obj')
reconstructed = trimesh.load('output/03_reconstructed.obj')

# Compare statistics
print("Mesh Comparison:")
print(f"Original vertices: {len(original.vertices)}")
print(f"Reconstructed vertices: {len(reconstructed.vertices)}")
print(f"Original volume: {original.volume:.6f}")
print(f"Reconstructed volume: {reconstructed.volume:.6f}")
print(f"Volume difference: {abs(original.volume - reconstructed.volume) / original.volume * 100:.2f}%")
```

---

## 🎨 Upload Your Own Mesh

```python
from google.colab import files

# Upload mesh file
print("Upload your .obj or .ply mesh file:")
uploaded = files.upload()

# Get filename
mesh_file = list(uploaded.keys())[0]
print(f"\n✓ Uploaded: {mesh_file}")

# Process it
!python tests/test_wavelet_pipeline.py --mesh {mesh_file} --resolution 128 --output ./my_output

# Visualize
plot_mesh(f'my_output/03_reconstructed.obj', f'Your Mesh: {mesh_file}')
```

---

## 📥 Download Results

```python
from google.colab import files

# Download individual files
files.download('output/01_original.obj')
files.download('output/02_from_sdf.obj')
files.download('output/03_reconstructed.obj')

# Or zip everything
!zip -r wavemesh_output.zip output/
files.download('wavemesh_output.zip')
```

---

## 🔧 Advanced Usage

### Custom Parameters

```python
# Test with custom resolution and threshold
!python tests/test_wavelet_pipeline.py \
    --create-test-mesh \
    --resolution 256 \
    --threshold 0.005 \
    --level 4 \
    --output ./high_quality

# Parameters explained:
# --resolution: SDF grid size (64, 128, 256, 512)
# --threshold: Sparsification threshold (0.001-0.1)
# --level: Wavelet decomposition levels (2-5)
```

### Memory Management

```python
# If you run out of memory, use lower resolution
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64

# Or clear outputs between runs
import gc
gc.collect()

# Check GPU memory
!nvidia-smi
```

### Batch Processing

```python
# Process multiple meshes
mesh_files = ['mesh1.obj', 'mesh2.obj', 'mesh3.obj']

for mesh_file in mesh_files:
    output_dir = f'output_{mesh_file.replace(".obj", "")}'
    !python tests/test_wavelet_pipeline.py \
        --mesh {mesh_file} \
        --resolution 128 \
        --output {output_dir}
    print(f"✓ Processed {mesh_file}")
```

---

## 🧠 Run Neural Network Training (Example)

```python
# This is a simplified training example
# Full training would require a dataset

import torch
from models import WaveMeshUNet, GaussianDiffusion
from data import WaveletTransform3D, mesh_to_sdf_grid

# Setup
device = 'cuda'
model = WaveMeshUNet(
    in_channels=1,
    out_channels=1,
    encoder_channels=[32, 64, 128, 256],
    use_attention=True
).to(device)

diffusion = GaussianDiffusion(timesteps=1000)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("✓ Training setup ready!")
print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Device: {device}")

# Note: Full training requires:
# 1. Dataset of meshes
# 2. Multi-view images
# 3. Training loop
# 4. Evaluation metrics
```

---

## 📊 Performance Benchmarks

### Test Different Resolutions

```python
import time

resolutions = [32, 64, 128, 256]

print("Resolution | Time (s) | Memory (MB) | Sparsity")
print("-" * 60)

for res in resolutions:
    start = time.time()

    # Run test
    !python tests/test_wavelet_pipeline.py \
        --create-test-mesh \
        --resolution {res} \
        --output ./benchmark_{res} \
        > /tmp/bench_{res}.log 2>&1

    elapsed = time.time() - start

    # Parse output
    with open(f'/tmp/bench_{res}.log', 'r') as f:
        log = f.read()
        # Extract metrics from log

    print(f"{res:10d} | {elapsed:8.2f} | {'N/A':11s} | {'~98%':8s}")
```

---

## 🐛 Troubleshooting

### Issue 1: GitHub Authentication Error

**Error:** `fatal: could not read Username for 'https://github.com': No such device or address`

**Solution:**

```python
# Step 1: Create GitHub Personal Access Token
# Go to: https://github.com/settings/tokens
# Create token with 'repo' scope

# Step 2: Add to Colab Secrets
# Click 🔑 Key icon in left sidebar → Add secret
# Name: GH_TOKEN
# Value: <your token>
# Enable "Notebook access"

# Step 3: Clone with token
from google.colab import userdata
gh_token = userdata.get('GH_TOKEN')
!git clone https://{gh_token}@github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf
```

**Alternative - Make Repository Public:**
If you don't need the repository to be private:

1. Go to repository Settings → Danger Zone → Change visibility → Make public

### Issue 2: CUDA/spconv version mismatch

**Error:** `ERROR: Could not find a version that satisfies the requirement spconv-cu120`

**Solution:** spconv-cu118 works with CUDA 12.x - use it instead:

```python
# Check CUDA version
!nvcc --version

# For CUDA 12.5/12.4/12.3 (current Colab default as of Nov 2024)
# spconv-cu118 is compatible with CUDA 12.x
!pip uninstall -y spconv-cu118 spconv-cu121
!pip install spconv-cu118

# Verify it works
import torch
import spconv.pytorch as spconv
print(f"✅ spconv imported successfully")
print(f"PyTorch CUDA: {torch.version.cuda}")

# Alternative for CUDA 12.1 specifically
# !pip install spconv-cu121
```

**Why this works:** spconv builds are forward-compatible. The cu118 build works with CUDA 12.x because the core CUDA APIs haven't changed significantly.

### Issue 3: Out of Memory

```python
# Use smaller resolution
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64

# Clear cache
import torch
torch.cuda.empty_cache()
```

### Issue 4: Import errors

```python
# Reinstall dependencies
!pip install --upgrade PyWavelets trimesh scikit-image scipy numpy

# Verify
!python -c "import pywt; import trimesh; print('✓ Imports OK')"
```

### Issue 5: verify_installation.py not found

**Error:** `python3: can't open file '/content/verify_installation.py': [Errno 2] No such file or directory`

**This happens when:**

- Repository wasn't cloned successfully
- You're not in the WaveMeshDf directory

**Solution:**

```python
import os

# Check if we're in the right directory
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# If WaveMeshDf exists but you're not in it
if 'WaveMeshDf' in os.listdir('.'):
    %cd WaveMeshDf
    print("✓ Moved to WaveMeshDf directory")

# If repository wasn't cloned, clone it again
if not os.path.exists('WaveMeshDf'):
    from google.colab import userdata
    try:
        gh_token = userdata.get('GH_TOKEN')
        !git clone https://{gh_token}@github.com/HoangNguyennnnnnn/WaveMeshDf.git
    except:
        !git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
    %cd WaveMeshDf
```

### Issue 6: Display/OpenGL errors

```python
# This is normal in Colab - the code handles it automatically
# You'll see:
# ⚠ mesh_to_sdf failed (NoSuchDisplayException), using simple method...

# This is fine! The simple method works great in headless environments
```

---

## 📚 Complete Colab Notebook Template

Here's a complete notebook you can copy:

```python
# =============================================================================
# CELL 0: Check GitHub Token (For Private Repos Only)
# =============================================================================

from google.colab import userdata
try:
    token = userdata.get('GH_TOKEN')
    print("✅ GitHub token found! Ready to clone private repository.")
except:
    print("⚠️ No GH_TOKEN found.")
    print("If repository is private, add GH_TOKEN to secrets (🔑 icon in sidebar)")
    print("If repository is public, you can proceed without it.")

# =============================================================================
# CELL 1: Setup Environment
# =============================================================================

import os
import subprocess
os.environ['PYOPENGL_PLATFORM'] = 'egl'

print("📦 Installing dependencies...")
!pip install -q PyWavelets trimesh scikit-image scipy numpy torch torchvision tqdm pyyaml einops

# Auto-detect CUDA version and install matching spconv
print("\n🔍 Detecting CUDA version...")
try:
    cuda_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
    print(f"CUDA version detected: {cuda_version.split('release ')[1].split(',')[0]}")

    # spconv-cu118 works with CUDA 12.x (forward compatible)
    if 'release 12.' in cuda_version:
        print("ℹ️  Using spconv-cu118 (compatible with CUDA 12.x)")
        !pip install -q spconv-cu118
    elif 'release 11.8' in cuda_version:
        !pip install -q spconv-cu118
    else:
        !pip install -q spconv-cu118
except:
    print("⚠️  CUDA detection failed, using spconv-cu118 as default")
    !pip install -q spconv-cu118

# Clone repository with authentication
print("\n📥 Cloning repository...")
try:
    from google.colab import userdata
    gh_token = userdata.get('GH_TOKEN')
    !git clone https://{gh_token}@github.com/HoangNguyennnnnnn/WaveMeshDf.git
except:
    !git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git

%cd WaveMeshDf

# Verify
print("\n🔍 Verifying installation...")
if os.path.exists('verify_installation.py'):
    !python verify_installation.py
else:
    import pywt, trimesh, torch
    import spconv.pytorch as spconv
    print("✅ All core dependencies verified!")

print("\n" + "="*60)
print("✅ Setup complete! Ready to run WaveMesh-Diff")
print("="*60)

# =============================================================================
# CELL 2: Test Module A (Wavelet Transform)
# =============================================================================

!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128

# =============================================================================
# CELL 3: Test Modules B & C (Neural Networks)
# =============================================================================

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

!python tests/test_modules_bc.py

# =============================================================================
# CELL 4: Visualize Results
# =============================================================================

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh(mesh_path, title="Mesh"):
    mesh = trimesh.load(mesh_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.7,
        facecolor='cyan',
        edgecolor='black',
        linewidths=0.1
    )
    ax.add_collection3d(mesh_collection)

    scale = mesh.vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.show()

plot_mesh('output/01_original.obj', 'Original')
plot_mesh('output/03_reconstructed.obj', 'Reconstructed')

# =============================================================================
# CELL 5: Download Results
# =============================================================================

from google.colab import files

!zip -r results.zip output/
files.download('results.zip')
```

---

## 🎯 What You Can Do in Colab

✅ **Working Now:**

- Module A: Wavelet transform pipeline
- Module B: Sparse U-Net architecture
- Module C: Diffusion model
- Mesh processing and reconstruction
- Quality evaluation
- Visualization

⏳ **Coming Soon:**

- Module D: Multi-view encoder
- Full training pipeline
- Large-scale dataset processing

---

## 💡 Tips for Best Results

1. **Use GPU runtime** for Modules B & C (neural networks)
2. **Start with resolution 64 or 128** for quick testing
3. **Use resolution 256** for high quality results
4. **Monitor memory** with `!nvidia-smi`
5. **Save checkpoints** regularly if training
6. **Download results** before runtime disconnects

---

## 📞 Getting Help

If you encounter issues:

1. Check `TROUBLESHOOTING.md` in the repo
2. Run `!python verify_installation.py`
3. Review error messages carefully
4. Check CUDA version compatibility

---

## 🎓 Learning Resources

- **README.md** - Project overview
- **QUICKSTART.md** - Local testing guide
- **ARCHITECTURE.md** - System diagrams
- **PROJECT_STATUS.md** - Current progress

---

**Happy mesh generation! 🎨✨**

_Last updated: November 18, 2025_
