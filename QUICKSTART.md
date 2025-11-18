# Quick Start Guide - Module A Testing

## Prerequisites

```bash
pip install -r requirements.txt
```

## Test 1: Simple Sphere Test (Recommended First Test)

This creates a test mesh and verifies the complete pipeline:

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 128 --output ./test_output
```

**What this does:**

1. Creates a simple sphere mesh (`test_mesh.obj`)
2. Converts mesh → SDF grid (128³)
3. Transforms SDF → Sparse wavelet coefficients
4. Reconstructs Sparse → SDF → Mesh
5. Saves 3 meshes to `./test_output/`:
   - `01_original.obj` - Input mesh
   - `02_from_sdf.obj` - Baseline (no compression)
   - `03_reconstructed.obj` - From sparse wavelet

**Expected results:**

- Sparsity: ~98-99%
- Compression: ~100-200x
- MSE: < 0.001
- All 3 meshes should look nearly identical

## Test 2: Higher Resolution Test

For better quality (but slower):

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 256
```

## Test 3: Test Your Own Mesh

```bash
python tests/test_wavelet_pipeline.py --mesh path/to/your/mesh.obj --resolution 128
```

## Test 4: Find Optimal Threshold

This tests multiple threshold values to find the best quality/compression tradeoff:

```bash
python tests/test_wavelet_pipeline.py --mesh path/to/mesh.obj --test-thresholds --resolution 128
```

**Typical output:**

```
Threshold    Sparsity     Compression     MSE         MAE
----------------------------------------------------------------------
0.0010       95.23%       20.95          0.000124    0.003456
0.0050       97.45%       39.22          0.000389    0.006712
0.0100       98.67%       75.19          0.000812    0.011234  ← Recommended
0.0200       99.12%       113.64         0.001567    0.018945
0.0500       99.56%       227.27         0.004321    0.035678
0.1000       99.78%       454.55         0.012456    0.067890
```

## Understanding the Output

### Console Output

```
[Step 1] Loading original mesh...
  ✓ Loaded mesh: 2562 vertices, 5120 faces

[Step 2] Converting mesh to SDF grid (resolution=128^3)...
  ✓ SDF grid shape: (128, 128, 128)
  ✓ SDF range: [-0.5234, 0.4891]
  ✓ Memory: 8.00 MB

[Step 3] Converting to sparse wavelet (level=3, threshold=0.01)...
  ✓ Sparse representation created
    - Total elements: 2,097,152
    - Non-zero elements: 27,841
    - Sparsity ratio: 98.67%
    - Compression ratio: 75.32x
    - Memory (dense): 8.00 MB
    - Memory (sparse): 0.11 MB

[Step 4] Reconstructing dense SDF from sparse wavelet...
  ✓ Reconstructed SDF shape: (128, 128, 128)

  Reconstruction Quality:
    - MSE: 0.000812
    - MAE: 0.011234
    - Max Error: 0.045678

[Step 7] Comparing meshes...
  Original Mesh (from SDF):
    - Vertices: 8234
    - Faces: 16464
    - Volume: 0.523456
    - Watertight: True

  Reconstructed Mesh:
    - Vertices: 8156
    - Faces: 16308
    - Volume: 0.521234
    - Watertight: True

  Volume difference: 0.42%

  Geometric Distance:
    - Approximate Hausdorff: 0.003456
    - Mean distance: 0.001234
```

### Quality Indicators

✅ **Good Quality:**

- MSE < 0.001
- Sparsity > 95%
- Volume difference < 2%
- Mean geometric distance < 0.01
- Watertight meshes

⚠️ **May need adjustment:**

- MSE > 0.005 → Reduce threshold
- Sparsity < 90% → Increase threshold or decomposition level
- Volume difference > 5% → Check mesh quality or reduce threshold

## Visualizing Results

You can open the output meshes in any 3D viewer:

- **MeshLab**: Free, cross-platform
- **Blender**: Full 3D suite
- **Online**: https://3dviewer.net/

Compare all three files side-by-side to verify quality.

## Common Parameters

### Resolution (`--resolution`)

- **64**: Fast testing (< 1 second)
- **128**: Good for development (2-5 seconds) ⭐
- **256**: High quality (10-30 seconds)
- **512**: Very high quality (1-3 minutes, needs ~500MB RAM)

### Wavelet Level (`--level`)

- **2**: Minimal compression
- **3**: Good balance ⭐
- **4**: Higher compression, may lose fine details
- **5**: Maximum compression, only for very smooth shapes

### Threshold (`--threshold`)

- **0.001**: Minimal compression, best quality
- **0.01**: Recommended balance ⭐
- **0.05**: High compression, acceptable quality
- **0.1**: Very high compression, may have artifacts

## Troubleshooting

### Import Error: No module named 'spconv'

```bash
# Install for your CUDA version
pip install spconv-cu118  # CUDA 11.8
# or
pip install spconv-cu121  # CUDA 12.1
```

### Import Error: No module named 'mesh_to_sdf'

```bash
pip install mesh-to-sdf
```

### Poor reconstruction quality (high MSE)

- Try reducing threshold: `--threshold 0.005`
- Increase resolution: `--resolution 256`
- Check if input mesh is watertight

### Out of memory

- Reduce resolution: `--resolution 64`
- Use lower decomposition level: `--level 2`

## Next Steps

Once Module A tests pass successfully:

1. ✅ Verify compression ratios are acceptable (>50x)
2. ✅ Verify reconstruction quality (MSE < 0.001)
3. ✅ Test with your own mesh dataset
4. ➡️ Ready to proceed to Module B: Sparse 3D U-Net

## Python API Usage

```python
# Quick example in a Jupyter notebook or script
from data.wavelet_utils import (
    WaveletTransform3D,
    mesh_to_sdf_grid,
    sparse_to_mesh,
    save_mesh
)

# Initialize
transformer = WaveletTransform3D(wavelet='bior4.4', level=3)

# Pipeline
sdf = mesh_to_sdf_grid("bunny.obj", resolution=128)
sparse = transformer.dense_to_sparse_wavelet(sdf, threshold=0.01)
vertices, faces = sparse_to_mesh(sparse)
save_mesh(vertices, faces, "bunny_reconstructed.obj")

# Check quality
from data.wavelet_utils import compute_sparsity
stats = compute_sparsity(sparse)
print(f"Compression: {stats['compression_ratio']:.1f}x")
```

## Performance Tips

1. **Use NumPy arrays** when not training (faster than torch tensors)
2. **Cache SDF grids** for repeated experiments
3. **Batch processing**: Process multiple meshes in parallel
4. **GPU**: Module A is CPU-only, but Modules B/C will use GPU

---

**Ready to test?** Run the first command above and verify you get good results! 🚀
