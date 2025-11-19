# Arguments Quick Reference

## 🔑 Key Difference

| Script        | Argument            | Purpose                 |
| ------------- | ------------------- | ----------------------- |
| `train.py`    | `--diffusion_steps` | Total timesteps (1000)  |
| `generate.py` | `--num_steps`       | Sampling steps (50-250) |

---

## Training

```bash
# Colab Free
python train.py --data_root data/ModelNet40 --resolution 16 --batch_size 4 --diffusion_steps 100

# Local GPU
python train.py --data_root data/ModelNet40 --resolution 32 --batch_size 16 --diffusion_steps 1000 --mixed_precision
```

**Key args:**

- `--resolution` - 16 (fast) | 32 (balanced) | 64 (quality)
- `--diffusion_steps` - 100 (debug) | 1000 (best)
- `--use_ema` - Better quality
- `--mixed_precision` - 2x faster

---

## Generation

```bash
# Fast
python generate.py --checkpoint outputs/best.pt --num_steps 50 --num_samples 10

# Quality
python generate.py --checkpoint outputs/best.pt --num_steps 250 --num_samples 3
```

**Key args:**

- `--num_steps` - 50 (fast) | 100 (balanced) | 250 (quality)

---

## Common Mistakes

❌ `generate.py --diffusion_steps 50` → ✅ `generate.py --num_steps 50`

❌ `train.py --num_steps 1000` → ✅ `train.py --diffusion_steps 1000`

---

**See:** [TRAINING.md](TRAINING.md) | [QUICKSTART.md](QUICKSTART.md)
