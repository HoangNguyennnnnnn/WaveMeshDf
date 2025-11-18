# Training Guide - WaveMesh-Diff

Complete guide for training WaveMesh-Diff models.

---

## 📋 Prerequisites

1. **Install Dependencies:**

```bash
pip install torch torchvision numpy
pip install PyWavelets trimesh matplotlib tqdm pyyaml
pip install transformers huggingface_hub  # Optional: DINOv2
pip install spconv-cu118  # Optional: GPU acceleration
```

2. **Download Dataset:**

```bash
# ModelNet40 (500MB - recommended for starting)
python scripts/download_data.py --dataset modelnet40

# Or ShapeNet (50GB - for production)
# Manual download from https://shapenet.org/
```

---

## 🚀 Quick Start

### Option 1: Using Config Files

```bash
# Debug mode (fast, for testing)
python train.py \
    --data_root data/ModelNet40 \
    --output_dir outputs/debug \
    --debug \
    --max_samples 20

# Default training
python train.py \
    --data_root data/ModelNet40 \
    --output_dir outputs/default

# High resolution (requires GPU + lots of RAM)
python train.py \
    --dataset shapenet \
    --data_root data/ShapeNet \
    --resolution 64 \
    --batch_size 4 \
    --mixed_precision \
    --use_ema
```

### Option 2: Using YAML Config

```bash
python train.py --config configs/default.yaml
python train.py --config configs/high_res.yaml
python train.py --config configs/debug.yaml
```

---

## 📊 Monitor Training

Training logs are saved to `outputs/{exp_name}/`:

```
outputs/
└── modelnet40_res32_20241118_120000/
    ├── config.yaml           # Training config
    ├── train.log             # Text logs
    ├── metrics.jsonl         # Metrics (JSON lines)
    ├── summary.json          # Summary
    ├── checkpoint_epoch_5.pth
    ├── checkpoint_epoch_10.pth
    ├── best.pth              # Best validation model
    └── final.pth             # Final model
```

**View metrics:**

```python
import json

# Load metrics
with open('outputs/.../metrics.jsonl') as f:
    metrics = [json.loads(line) for line in f]

# Plot loss
import matplotlib.pyplot as plt
train_losses = [m['train/loss'] for m in metrics if 'train/loss' in m]
plt.plot(train_losses)
plt.show()
```

---

## 🎨 Generate Meshes

After training, generate new meshes:

```bash
python generate.py \
    --checkpoint outputs/.../best.pth \
    --num_samples 100 \
    --resolution 32 \
    --output_dir generated_meshes
```

Output:

```
generated_meshes/
├── mesh_0000.obj
├── mesh_0001.obj
├── ...
└── mesh_0099.obj
```

---

## ⚙️ Configuration Options

### Dataset

- `--dataset`: `modelnet40` or `shapenet`
- `--data_root`: Path to dataset
- `--resolution`: SDF resolution (16, 32, 64, 128)
- `--wavelet_threshold`: Sparsity threshold (0.001-0.1)

### Model

- `--unet_channels`: U-Net channels (e.g., `16 32 64 128`)
- `--time_emb_dim`: Time embedding dimension
- `--use_attention`: Enable self-attention
- `--context_dim`: Multi-view conditioning (0=unconditional)

### Training

- `--batch_size`: Batch size (2-16 depending on GPU)
- `--epochs`: Number of epochs
- `--lr`: Learning rate (1e-4 recommended)
- `--mixed_precision`: Enable FP16 training

### Advanced

- `--use_ema`: Exponential moving average (improves quality)
- `--cfg_prob`: Classifier-free guidance probability
- `--grad_clip`: Gradient clipping threshold

---

## 📈 Expected Results

### Debug Mode (16³, 20 samples)

- **Time:** ~5 minutes
- **Purpose:** Test pipeline

### ModelNet40 (32³, full dataset)

- **Time:** ~2-4 hours (RTX 3080)
- **Quality:** Basic shapes, good for prototyping

### ShapeNet (64³, full dataset)

- **Time:** ~1-2 days (RTX 3080)
- **Quality:** Production-quality meshes

---

## 🐛 Troubleshooting

### "CUDA out of memory"

```bash
# Reduce batch size
--batch_size 2

# Reduce resolution
--resolution 16

# Enable mixed precision
--mixed_precision
```

### "Dataset not found"

```bash
# Check data path
ls data/ModelNet40

# Re-download
python scripts/download_data.py --dataset modelnet40
```

### Training too slow

```bash
# Enable all optimizations
--mixed_precision \
--num_workers 4

# Or use smaller model
--unet_channels 8 16 32
```

### Poor generation quality

```bash
# Enable EMA
--use_ema

# More diffusion steps
--diffusion_steps 1000

# Lower wavelet threshold
--wavelet_threshold 0.005

# Use attention
--use_attention
```

---

## 📝 Tips

1. **Start small:** Use debug config first to verify pipeline
2. **Monitor metrics:** Check `train.log` and `metrics.jsonl`
3. **Use EMA:** Significantly improves quality
4. **Mixed precision:** 2x faster, same quality
5. **Checkpoints:** Save frequently in case of crashes

---

## 🔄 Resume Training

```bash
python train.py \
    --data_root data/ModelNet40 \
    --resume outputs/.../checkpoint_epoch_50.pth
```

---

## 📚 Next Steps

1. **Improve quality:** Add multi-view conditioning
2. **Scale up:** Train on ShapeNet with 64³ resolution
3. **Add metrics:** Implement Chamfer distance evaluation
4. **Deploy:** Export to ONNX or TorchScript

See [ROADMAP.md](ROADMAP.md) for more details.
