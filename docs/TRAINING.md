# Training Guide

## Quick Commands

**Colab (Auto-optimized):**
```bash
python train_colab.py --mode debug  # 5 min test
python train_colab.py --mode full   # Full training
```

**Manual:**
```bash
# Colab Free
python train.py --data_root data/ModelNet40 --resolution 16 --batch_size 4 --diffusion_steps 100

# Local GPU
python train.py --data_root data/ModelNet40 --resolution 32 --batch_size 16 --diffusion_steps 1000 --mixed_precision
```

---

## Setup

```bash
# Install
pip install -r requirements.txt

# Download data
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip -d data/
```

---

## Key Arguments

| Argument | Values | Notes |
|----------|--------|-------|
| `--resolution` | 16/32/64 | 16=fast, 32=balanced, 64=quality |
| `--diffusion_steps` | 100/500/1000 | Training timesteps |
| `--batch_size` | 4/8/16 | Depends on RAM |
| `--use_ema` | flag | Better quality |
| `--mixed_precision` | flag | 2x faster |

**Note:** Use `--diffusion_steps` for training, `--num_steps` for generation!

---

## Generate After Training

```bash
# Fast (50 steps)
python generate.py --checkpoint outputs/best.pt --num_steps 50 --num_samples 10

# Quality (250 steps)
python generate.py --checkpoint outputs/best.pt --num_steps 250 --num_samples 5
```

---

## Common Issues

**OOM Error:**
```bash
--resolution 16 --batch_size 4 --num_workers 0
```

**Slow Training:**
```bash
--mixed_precision --diffusion_steps 100
```

**Poor Quality:**
```bash
--use_ema --diffusion_steps 1000 --wavelet_threshold 0.001
```

---

## Resume Training

```bash
python train.py --resume outputs/checkpoint_latest.pt
```

---

**See:** [ARGUMENTS_REFERENCE.md](ARGUMENTS_REFERENCE.md) | [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
