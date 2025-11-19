# WaveMesh-Diff Architecture

## System Overview

```
Multi-view Images → DINOv2 Encoder → Diffusion Model → Sparse 3D U-Net → Wavelet Transform → 3D Mesh
```

## Modules

| Module       | Status      | Description                                   |
| ------------ | ----------- | --------------------------------------------- |
| **Module A** | ✅ Complete | Wavelet transform (SDF ↔ sparse coefficients) |
| **Module B** | ⏳ Future   | Sparse 3D U-Net (denoising network)           |
| **Module C** | ⏳ Future   | Diffusion model (DDPM/DDIM)                   |
| **Module D** | ⏳ Future   | Multi-view encoder (DINOv2)                   |

## Module A: Wavelet Pipeline (Implemented)

### Data Flow

```
Mesh (.obj) → SDF (256³, 64MB) → Sparse Wavelet (160K coeffs, 1.3MB) → Reconstructed SDF → Mesh
```

### Wavelet Structure (3 levels)

- **Level 1**: 128³ × 8 subbands (1 approx + 7 detail)
- **Level 2**: 64³ × 8 subbands
- **Level 3**: 32³ × 8 subbands
- **Total**: 24 channels in sparse representation

### Sparsification

- **Threshold**: |coeff| < 0.01 removed
- **Sparsity**: ~99% (16M → 160K coefficients)
- **Compression**: 75x smaller
- **Quality**: MSE < 0.0001

### Key Functions

```python
# Convenience pipeline
mesh_to_sdf_simple(mesh_path, resolution=256)
sdf_to_sparse_wavelet(sdf, threshold=0.01)
sparse_wavelet_to_sdf(sparse_data)
sdf_to_mesh(sdf)
```

## Training Pipeline (Future: Modules B+C+D)

```
1. Load mesh → sparse wavelet (Module A)
2. Load images → features (Module D: DINOv2)
3. Add noise to coeffs (Module C: forward diffusion)
4. Denoise with U-Net (Module B: sparse 3D U-Net + cross-attention)
5. Compute MSE loss
6. Backprop & update
```

## Inference Pipeline (Future)

```
1. Multi-view images → DINOv2 features
2. Sample noise: z_T ~ N(0,I)
3. DDIM sampling: z_T → z_0 (using U-Net)
4. z_0 → SDF → Mesh (Module A)
```

## File Structure

```
data/wavelet_utils.py      ⭐ Module A implementation
tests/test_wavelet_*.py    ⭐ Tests
examples/colab_quickstart  ⭐ Demo notebook
models/                    Future: U-Net & Diffusion
utils/                     Future: Helpers
```

## Quality Metrics

| Configuration                   | Sparsity | MSE   | Quality      |
| ------------------------------- | -------- | ----- | ------------ |
| High (bior4.4, L3, thresh=1e-4) | 90%      | <1e-4 | 🟢 Excellent |
| Balanced (db2, L2, thresh=1e-3) | 95%      | <1e-3 | 🟢 Good      |
| Lossless (thresh=0)             | 70%      | ≈0    | 🟢 Perfect   |

## Testing

```bash
python tests/test_wavelet_pipeline.py
```

**Output**:

- Original mesh, SDF mesh, reconstructed mesh
- Metrics: MSE, sparsity, compression ratio
- Result: ✅ PASS if MSE < 0.001
