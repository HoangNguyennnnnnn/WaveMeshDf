# 📋 Colab Training Commands Cheat Sheet

## ⚡ Recommended (Auto-Optimized)

```bash
# Quick test (5 min) - START HERE! ✅
!python train_colab.py --mode debug

# Medium test (15 min)
!python train_colab.py --mode quick

# Full training (3 hours)
!python train_colab.py --mode full
```

**Why use `train_colab.py`?**

- ✅ Auto-detects RAM/GPU
- ✅ Sets optimal resolution, batch size, workers
- ✅ Prevents OOM crashes
- ✅ Works on Colab free tier (12GB RAM)

---

## 🛠️ Manual Commands (Advanced)

### Colab Free Tier (12GB RAM)

```bash
!python train.py \
    --data_root data/ModelNet40 \
    --dataset modelnet40 \
    --resolution 16 \
    --batch_size 4 \
    --num_workers 0 \
    --unet_channels 8 16 32 \
    --diffusion_steps 100 \
    --epochs 5 \
    --max_samples 20 \
    --output_dir outputs/debug
```

### Colab Pro (25GB RAM)

```bash
!python train.py \
    --data_root data/ModelNet40 \
    --dataset modelnet40 \
    --resolution 32 \
    --batch_size 8 \
    --num_workers 1 \
    --unet_channels 16 32 64 \
    --diffusion_steps 500 \
    --epochs 20 \
    --output_dir outputs/full
```

### Local Machine (32GB+ RAM)

```bash
python train.py \
    --data_root data/ModelNet40 \
    --dataset modelnet40 \
    --resolution 32 \
    --batch_size 16 \
    --num_workers 4 \
    --unet_channels 16 32 64 128 \
    --diffusion_steps 1000 \
    --epochs 100 \
    --mixed_precision \
    --use_ema \
    --output_dir outputs/full_local
```

---

## 🔑 Key Parameters

### Memory-Critical (Avoid OOM)

```bash
--num_workers 0        # MOST IMPORTANT for Colab!
--resolution 16        # Lower = less RAM (8/16/24/32)
--batch_size 4         # Smaller = less RAM
--unet_channels 8 16   # Fewer/smaller channels
```

### Performance

```bash
--mixed_precision      # 2x faster, uses less VRAM
--num_workers 2        # Only if RAM > 16GB
```

### Quality

```bash
--diffusion_steps 1000 # More steps = better quality
--unet_channels 16 32 64 128  # Larger model
--resolution 32        # Higher resolution
```

### Training Length

```bash
--epochs 100           # Full training
--max_samples 20       # Quick debug test
```

---

## 📊 Presets by System

| System         | Resolution | Batch | Workers | Channels     | Steps | Epochs |
| -------------- | ---------- | ----- | ------- | ------------ | ----- | ------ |
| **Colab Free** | 16         | 4     | 0       | 8 16 32      | 100   | 5-20   |
| **Colab Pro**  | 32         | 8     | 1       | 16 32 64     | 500   | 20-50  |
| **Local 32GB** | 32         | 16    | 4       | 16 32 64 128 | 1000  | 100+   |

---

## 🚨 Troubleshooting

### Out of Memory (OOM)

```bash
# Try smallest settings
!python train.py \
    --resolution 8 \
    --batch_size 1 \
    --num_workers 0 \
    --unet_channels 4 8
```

### Too Slow (no GPU)

```bash
# Enable GPU in Colab:
# Runtime → Change runtime type → T4 GPU → Save
```

### Worker Killed Error

```bash
# Always use num_workers=0 on Colab!
!python train.py --num_workers 0 ...
```

---

## 📂 Output Structure

```
outputs/
  └── debug/
      └── modelnet40_res16_20231119_033649/
          ├── train.log           # Training logs
          ├── checkpoint_epoch_5.pth  # Model checkpoint
          ├── final.pth           # Final model
          └── config.yaml         # Training config
```

**Check results:**

```bash
!ls -lh outputs/debug/
!tail -20 outputs/debug/*/train.log
```

---

## 🎯 Quick Workflows

### 1. First Time Setup Test

```bash
# Just verify everything works (5 min)
!python train_colab.py --mode debug
```

### 2. Quick Prototype

```bash
# Test on 100 samples (15 min)
!python train_colab.py --mode quick
```

### 3. Full Training

```bash
# Complete training (3 hours)
!python train_colab.py --mode full --epochs 50
```

### 4. Resume Training

```bash
!python train.py \
    --resume outputs/full/final.pth \
    --epochs 100 \
    --output_dir outputs/full_continued
```

---

## 🔗 Related Docs

- [MEMORY_FIX.md](MEMORY_FIX.md) - Technical details of memory optimization
- [TRAINING.md](TRAINING.md) - Full training guide
- [COLAB_SETUP.md](COLAB_SETUP.md) - Setup and troubleshooting

---

**TL;DR:** Use `!python train_colab.py --mode debug` for automatic optimization! 🎉
