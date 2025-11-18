# 🚀 Quick Start Guide - WaveMesh-Diff

## Bắt Đầu Ngay trong 30 Phút

### Bước 1: Cài Đặt Dependencies (5 phút)

```bash
# Dependencies cơ bản (bắt buộc)
pip install PyWavelets numpy torch torchvision
pip install trimesh matplotlib tqdm pillow

# Dependencies cho rendering (nếu train với real data)
pip install pyrender

# Dependencies cho DINOv2 (tùy chọn - cải thiện quality)
pip install transformers huggingface_hub

# Dependencies cho GPU (tùy chọn - tăng tốc 10-100x)
# pip install spconv-cu118  # Thay cu118 bằng CUDA version của bạn
```

### Bước 2: Verify Installation (2 phút)

```bash
# Test tất cả modules
python test_all_modules.py

# Kỳ vọng: 4/4 modules PASS
# Nếu Module A fail: pip install PyWavelets
```

### Bước 3: Download Data (10 phút)

**Option A: ModelNet40 (Nhanh - 500MB)**

```bash
python scripts/download_data.py --dataset modelnet40

# Tự động download + extract
# Data sẽ ở: ./data/ModelNet40/
```

**Option B: ShapeNet (Chất lượng cao - 50GB)**

```bash
# Manual download (cần đăng ký)
python scripts/download_data.py --dataset shapenet

# Follow instructions to download từ shapenet.org
```

### Bước 4: Test Rendering (5 phút)

```bash
# Test rendering script
python scripts/render_multiview.py --test

# Render một mesh cụ thể
python scripts/render_multiview.py \
    --mesh data/ModelNet40/chair/train/chair_0001.off \
    --output test_renders/ \
    --num_views 8
```

### Bước 5: Train Your First Model (8 phút)

```bash
# Quick training test (overfit on 10 samples)
python train_simple.py \
    --data_root data/ModelNet40 \
    --category chair \
    --num_samples 10 \
    --num_epochs 50 \
    --batch_size 2

# Kỳ vọng: Loss giảm từ ~0.5 → ~0.05
```

---

## 📊 Kiểm Tra Kết Quả

### 1. Visualize Training

```bash
# Xem generated meshes
python visualize_results.py --checkpoint checkpoints/latest.pt

# Sẽ tạo file visualization.png với:
# - Input multi-view images
# - Generated mesh
# - Ground truth mesh
```

### 2. Evaluate Model

```bash
python evaluate.py \
    --checkpoint checkpoints/latest.pt \
    --data_root data/ModelNet40 \
    --category chair

# Metrics:
# - Chamfer Distance: ~0.005 (lower is better)
# - F-Score@0.01: ~0.85 (higher is better)
```

---

## 🎯 Next Steps

### Cải Thiện Quality

1. **Train lâu hơn**

   ```bash
   python train.py --num_epochs 500  # Thay vì 50
   ```

2. **Tăng resolution**

   ```bash
   python train.py --resolution 64  # Thay vì 32
   ```

3. **Dùng DINOv2**
   ```bash
   pip install transformers
   huggingface-cli login
   python train.py --use_dinov2  # Tự động load pretrained weights
   ```

### Train Trên Full Dataset

```bash
# Full ShapeNet chairs (~7K samples)
python train.py \
    --data_root data/ShapeNetCore.v2/03001627 \
    --num_epochs 200 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_ema \
    --mixed_precision
```

### Multi-GPU Training

```bash
# Distributed training (nếu có nhiều GPUs)
torchrun --nproc_per_node=4 train_distributed.py \
    --data_root data/ShapeNetCore.v2 \
    --batch_size 64
```

---

## 🐛 Troubleshooting

### "Out of Memory"

```bash
# Giảm batch size
python train.py --batch_size 2

# Hoặc dùng gradient accumulation
python train.py --batch_size 2 --accumulation_steps 8
```

### "ModuleNotFoundError: No module named 'pywt'"

```bash
pip install PyWavelets
```

### "CUDA out of memory"

```bash
# Giảm resolution
python train.py --resolution 16

# Hoặc chạy trên CPU (chậm)
python train.py --device cpu
```

### Rendering Fails (Headless Server)

```bash
# Install OSMesa
conda install -c conda-forge osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

## 📈 Expected Performance

### Training Time

| Setup        | Resolution | Batch Size | Time/Epoch | Hardware          |
| ------------ | ---------- | ---------- | ---------- | ----------------- |
| CPU          | 32³        | 2          | ~30 min    | i7 CPU            |
| GPU (Dense)  | 32³        | 8          | ~5 min     | RTX 3080          |
| GPU (Sparse) | 32³        | 16         | ~2 min     | RTX 3080 + spconv |
| GPU (Sparse) | 64³        | 8          | ~8 min     | RTX 3080 + spconv |

### Quality Benchmarks

| Epochs | Resolution | Chamfer Distance | F-Score | Visual Quality |
| ------ | ---------- | ---------------- | ------- | -------------- |
| 50     | 32³        | 0.02             | 0.65    | Low            |
| 100    | 32³        | 0.008            | 0.78    | Medium         |
| 200    | 64³        | 0.003            | 0.88    | Good           |
| 500    | 64³        | 0.001            | 0.93    | Excellent      |

---

## 💡 Tips

1. **Overfit First**: Train trên 10 samples để verify code đúng
2. **Monitor Closely**: Check generated samples mỗi 10 epochs
3. **Start Small**: Dùng resolution 16³ để iterate nhanh
4. **Save Often**: Checkpoint mỗi epoch (disk is cheap)
5. **Compare Baselines**: So sánh với Point-E, Shap-E

---

## 🎓 Learning Resources

- **ROADMAP.md** - Detailed roadmap và improvements
- **TEST_ALL_MODULES.md** - Testing documentation
- **PROJECT_EXPLANATION.md** - Vietnamese architecture explanation
- **ARCHITECTURE.md** - English technical details

---

## 📞 Need Help?

1. Check TROUBLESHOOTING.md
2. Run tests: `python test_all_modules.py`
3. Check GitHub Issues
4. Ask on forums (include error logs + config)

**Happy Training! 🚀**
