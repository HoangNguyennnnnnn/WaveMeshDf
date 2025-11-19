# Colab Setup Guide

## 📚 Which Notebook?

| Notebook | Time | Best For |
|----------|------|----------|
| [colab_quickstart.ipynb](https://colab.research.google.com/github/HoangNguyennnnnnn/WaveMeshDf/blob/main/colab_quickstart.ipynb) ⭐ | 15-20 min | Full demo |

---

## 🚀 Quick Setup

### 1. Enable GPU

**Critical for speed (10-50x faster)!**

1. **Runtime → Change runtime type**
2. **Hardware accelerator → T4 GPU**
3. **Save**

Verify:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### 2. RAM Limits

| Tier | RAM | Best Resolution |
|------|-----|-----------------|
| Free | 12GB | 32³ |
| Pro | 25GB | 64³ |

**Notebook auto-detects and uses safe resolution.**

### 3. Fix RAM Crashes

```python
# Clear memory
import gc; import torch
gc.collect()
torch.cuda.empty_cache()
```

Or: **Runtime → Restart runtime**

---

## 🎯 Training Configs

### Colab Free (12GB RAM)

```bash
python train.py --resolution 16 --batch_size 4 --diffusion_steps 100
```

### Colab Pro (25GB RAM)

```bash
python train.py --resolution 32 --batch_size 8 --diffusion_steps 500
```

---

## Common Issues

### "No GPU available"
- Enable GPU: Runtime → Change runtime type → T4 GPU

### "Runtime crashed"
- Reduce resolution: `--resolution 16`
- Reduce batch: `--batch_size 4`
- Restart runtime

### Slow training
- Enable GPU (see step 1)
- Use `--mixed_precision`

---

**See:** [QUICKSTART.md](QUICKSTART.md) | [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
