# Google Colab Setup for WaveMesh-Diff

**Now supports all modules (A, B, C) in Colab!** 🎉

## Quick Setup (Copy this cell to your Colab notebook)

```python
# Install dependencies for all modules
!pip install -q PyWavelets trimesh scikit-image scipy numpy torch torchvision rtree

# Configure for headless environment
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Clone repository
!git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf

# Run Module A test (Wavelet Transform) - Full speed!
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64

# Note: Modules B & C use dense fallback mode (slower but functional)
# No spconv required! Works great for testing and learning.
```

## Performance Note

- **Module A (Wavelet)**: Full speed ⚡
- **Modules B & C (Neural Networks)**: Dense fallback mode 🐢
  - ~10-50x slower than native spconv
  - Perfect for testing, learning, prototyping
  - For production: Use local GPU with spconv

The dense fallback automatically activates when spconv is not available!

## What to Expect

The test script will automatically:

1. Create a simple test sphere mesh
2. Convert to SDF using the simple method (works in Colab)
3. Transform to sparse wavelet representation
4. Reconstruct and compare quality

**Note**: In Colab, the simple SDF method is used by default. This is:

- ✅ 100% compatible with headless environments
- ✅ 10-100x faster than scan-based method
- ✅ Sufficient quality for wavelet testing
- ⚠️ Slightly less accurate for complex geometry

## Expected Output

```
Creating simple test sphere at ./test_mesh.obj...
✓ Test mesh created: 642 vertices, 1280 faces
================================================================================
WaveMesh-Diff Wavelet Pipeline Test
================================================================================

[Step 1] Loading original mesh...
  ✓ Loaded mesh: 642 vertices, 1280 faces

[Step 2] Converting mesh to SDF grid (resolution=64^3)...
  ⚠ mesh_to_sdf failed (NoSuchDisplayException), using simple method...
  ✓ SDF grid shape: (64, 64, 64)

[Step 3] Converting to sparse wavelet...
  ✓ Sparsity ratio: 97.2%
  ✓ Compression ratio: 35.71x

[Step 4] Reconstructing dense SDF...
  Reconstruction Quality:
    - MSE: 0.001234
    - MAE: 0.012345

✓ All tests completed successfully!
```

## View Output Meshes

```python
# List output files
!ls -lh output/

# Download files (optional)
from google.colab import files

# Download original mesh
files.download('output/01_original.obj')

# Download reconstructed mesh
files.download('output/03_reconstructed.obj')
```

## Visualize in Colab

```python
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh(mesh_path, title):
    mesh = trimesh.load(mesh_path)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh
    mesh_collection = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.7,
        facecolor='cyan',
        edgecolor='black'
    )
    ax.add_collection3d(mesh_collection)

    # Set limits
    scale = mesh.vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_title(title)
    plt.show()

# Plot original
plot_mesh('output/01_original.obj', 'Original Mesh')

# Plot reconstructed
plot_mesh('output/03_reconstructed.obj', 'Reconstructed from Sparse Wavelet')
```

## Advanced: Test with Different Parameters

```python
# Higher resolution (slower but better quality)
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128

# Test different thresholds
!python tests/test_wavelet_pipeline.py --create-test-mesh --test-thresholds --resolution 64

# Custom threshold
!python tests/test_wavelet_pipeline.py --create-test-mesh --threshold 0.005 --resolution 64
```

## Troubleshooting

### "Module not found" errors

```python
!pip install -r requirements.txt
```

### Want to use your own mesh?

```python
# Upload mesh file
from google.colab import files
uploaded = files.upload()

# Test with your mesh
!python tests/test_wavelet_pipeline.py --mesh your_mesh.obj --resolution 64
```

### Check installation

```python
!python verify_installation.py
```

## Performance Tips for Colab

1. **Start with low resolution** (64³) for fast testing
2. **Use simple SDF method** (automatic in Colab)
3. **Cache results** if processing multiple meshes
4. **Free RAM** between runs: Runtime → Restart runtime

## Next Steps

Once tests pass:

- ✅ Module A works at full speed!
- ✅ Modules B & C work in dense fallback mode
- ✅ Test neural network architectures
- ✅ Experiment with diffusion models
- ✅ Test with your own meshes
- ➡️ For optimal performance: Use local GPU setup with spconv

---

## 📊 Performance Comparison

| Module   | Colab (Dense Fallback) | Local (with spconv) |
| -------- | ---------------------- | ------------------- |
| Module A | ⚡ Full speed          | ⚡ Full speed       |
| Module B | 🐢 10-50x slower       | ⚡ Optimal          |
| Module C | 🐢 10-50x slower       | ⚡ Optimal          |

**Recommendation**: Colab is great for learning and testing. For production or large-scale experiments, use local GPU with spconv.

---

_This Colab setup now supports all three modules! Enjoy the complete WaveMesh-Diff pipeline! 🎨✨_
