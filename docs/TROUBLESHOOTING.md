# Troubleshooting

## Training Issues

### Out of Memory (OOM)

**Symptoms:** "Runtime crashed", "DataLoader worker killed"

**Fix:**
```bash
# Use smaller resolution
python train.py --resolution 16 --batch_size 4 --num_workers 0

# Or auto-optimize
python train_colab.py --mode debug
```

**RAM limits:**
- Colab Free (12GB) → `resolution=16`
- Colab Pro (25GB) → `resolution=32`

---

### No GPU Available

**Symptom:** Training very slow

**Fix:**
1. Runtime → Change runtime type
2. Hardware accelerator → T4 GPU
3. Save

---

### High Wavelet MSE

**Symptom:** Reconstruction quality poor (MSE > 0.001)

**Fix:** Already improved! Default settings give MSE < 0.00001

```python
# Good (default)
sparse = sdf_to_sparse_wavelet(sdf, threshold=0.01)

# Better  
sparse = sdf_to_sparse_wavelet(sdf, threshold=0.001)

# Lossless
sparse = sdf_to_sparse_wavelet(sdf, lossless=True)
```

---

### Module Import Errors

**Fix:**
```bash
# Install dependencies
pip install -r requirements.txt

# Restart runtime
# Runtime → Restart runtime
```

---

### Wrong Arguments

**Common mistakes:**

❌ `generate.py --diffusion_steps 50`  
✅ `generate.py --num_steps 50`

❌ `train.py --num_steps 1000`  
✅ `train.py --diffusion_steps 1000`

See: [ARGUMENTS_REFERENCE.md](ARGUMENTS_REFERENCE.md)

---

### Dataset Not Found

**Fix:**
```bash
# Download ModelNet40
python scripts/download_data.py --dataset modelnet40

# Or manual
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip -d data/
```

---

### Training Too Slow

**Fix:**
```bash
# Enable all optimizations
python train.py --mixed_precision --use_ema

# Or use smaller model
python train.py --unet_channels 8 16 32 --diffusion_steps 100
```

---

## Quick Fixes

| Issue | Solution |
|-------|----------|
| OOM | `--resolution 16 --batch_size 4` |
| No GPU | Runtime → Change runtime → T4 GPU |
| Slow | `--mixed_precision` |
| Import error | `pip install -r requirements.txt` |
| Dataset error | Re-download with script |

---

**See:** [COLAB_SETUP.md](COLAB_SETUP.md) | [TRAINING.md](TRAINING.md)
