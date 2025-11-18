# Troubleshooting Guide - WaveMesh-Diff

## Common Issues and Solutions

### 1. Display / OpenGL Errors (Headless Environments)

**Error:**

```
pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"
```

**Solution:**
This is automatically handled now! The code falls back to a simple SDF method. You'll see:

```
⚠ mesh_to_sdf failed (NoSuchDisplayException), using simple method...
```

**Manual fix (if needed):**

```python
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
```

Or run:

```bash
python setup_headless.py
```

---

### 2. Module Import Errors

**Error:**

```
ModuleNotFoundError: No module named 'pywt'
```

**Solution:**

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install PyWavelets trimesh scikit-image scipy numpy
```

---

### 3. Spconv Installation Issues

**Error:**

```
Could not find a version that satisfies the requirement spconv-cu118
```

**Solution:**
Match your CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Install matching spconv
pip install spconv-cu118  # For CUDA 11.8
pip install spconv-cu121  # For CUDA 12.1
```

**Note**: spconv is only needed for Module B (U-Net), not for Module A testing.

---

### 4. Poor Reconstruction Quality

**Issue:**
MSE > 0.005 or meshes look different

**Solutions:**

1. **Lower the threshold:**

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --threshold 0.005
```

2. **Increase resolution:**

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 256
```

3. **Use more decomposition levels:**

```python
transformer = WaveletTransform3D(wavelet='bior4.4', level=4)
```

---

### 5. Out of Memory

**Error:**

```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce resolution:**

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64
```

2. **Use higher threshold (more sparse):**

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --threshold 0.05
```

3. **Check memory usage:**

```python
from data import compute_sparsity
stats = compute_sparsity(sparse_data)
print(f"Memory: {stats['memory_sparse_mb']:.2f} MB")
```

---

### 6. Mesh Loading Errors

**Error:**

```
ValueError: Mesh file not found
```

**Solution:**
Use absolute paths or check file exists:

```python
import os
mesh_path = os.path.abspath("mesh.obj")
assert os.path.exists(mesh_path), f"File not found: {mesh_path}"
```

---

### 7. Google Colab Specific Issues

**Issue:**
Runtime disconnects or slow performance

**Solutions:**

1. **Start with smaller resolution:**

```bash
!python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64
```

2. **Free memory between runs:**

```python
# In Colab: Runtime → Restart runtime
import gc
gc.collect()
```

3. **Use simple SDF method (already default):**
   The code automatically detects Colab and uses the fast method.

---

### 8. Wavelet Transform Errors

**Error:**

```
ValueError: Invalid wavelet name
```

**Solution:**
Use supported wavelets:

```python
# Recommended
transformer = WaveletTransform3D(wavelet='bior4.4', level=3)

# Other options
# 'db4', 'coif3', 'sym8', 'haar'
```

---

### 9. Marching Cubes Issues

**Error:**

```
RuntimeError: No surface found
```

**Possible causes:**

- SDF values are all same sign
- Resolution too low
- Level set value incorrect

**Solutions:**

1. **Check SDF range:**

```python
print(f"SDF range: [{sdf_grid.min()}, {sdf_grid.max()}]")
# Should have both positive and negative values
```

2. **Adjust level:**

```python
vertices, faces = sdf_to_mesh(sdf_grid, level=0.0)
```

3. **Increase resolution:**

```python
sdf_grid = mesh_to_sdf_grid(mesh_path, resolution=256)
```

---

### 10. Test Script Not Running

**Error:**

```
python: can't open file 'tests/test_wavelet_pipeline.py'
```

**Solution:**
Check you're in the correct directory:

```bash
cd WaveMesh-Diff
ls tests/test_wavelet_pipeline.py  # Should exist
python tests/test_wavelet_pipeline.py --create-test-mesh
```

---

## Performance Tips

### Speed Optimization

1. **Lower resolution for testing:**

   - 64³: ~1 second
   - 128³: ~5 seconds
   - 256³: ~30 seconds

2. **Use simple SDF method:**

   - 10-100x faster than scan method
   - Automatic in headless environments

3. **Cache SDF grids:**

```python
import numpy as np

# Save
np.save('sdf_cache.npy', sdf_grid)

# Load
sdf_grid = np.load('sdf_cache.npy')
```

### Memory Optimization

1. **Increase threshold:**

   - threshold=0.01: ~99% sparse
   - threshold=0.05: ~99.5% sparse

2. **Process in batches:**

```python
# Instead of one large grid, use multiple smaller ones
for i in range(num_batches):
    sdf_patch = ...
    sparse_patch = transformer.dense_to_sparse_wavelet(sdf_patch)
```

---

## Verification Checklist

Run this to verify everything works:

```bash
# 1. Check installation
python verify_installation.py

# 2. Run basic test (fast)
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64

# 3. Check outputs exist
ls output/

# 4. Test quality (slower)
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128 --test-thresholds
```

**Expected results:**

- ✅ All imports successful
- ✅ MSE < 0.01 (good quality)
- ✅ Sparsity > 95%
- ✅ 3 mesh files in output/

---

## Still Having Issues?

1. **Check Python version:**

```bash
python --version  # Should be 3.9+
```

2. **Check dependencies:**

```bash
pip list | grep -i "pywavelets\|trimesh\|scikit-image"
```

3. **Run verification script:**

```bash
python verify_installation.py
```

4. **Check example code:**

```python
from data import WaveletTransform3D
import numpy as np

# Minimal test
test_sdf = np.random.randn(32, 32, 32).astype(np.float32)
transformer = WaveletTransform3D(wavelet='bior4.4', level=2)
sparse = transformer.dense_to_sparse_wavelet(test_sdf, threshold=0.1)
print("✓ Basic functionality works!")
```

---

## Environment-Specific Guides

- **Google Colab**: See [COLAB_SETUP.md](COLAB_SETUP.md)
- **Remote Server**: Run `python setup_headless.py`
- **Windows**: Should work out of the box
- **Linux**: May need `sudo apt-get install python3-dev`
- **macOS**: Use `brew install python3`

---

**Last Updated:** November 18, 2025
