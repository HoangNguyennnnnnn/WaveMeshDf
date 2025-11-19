# 🚀 Quick Start - WaveMesh-Diff (Updated for Memory Fix)

Bắt đầu sử dụng WaveMesh-Diff trong 10-30 phút.

---

## ⚡ NEW: Memory-Optimized Training

**🔥 Lỗi OOM đã được fix!** Dùng lệnh mới này:

```bash
# In Colab - Auto-optimized, no crashes!
!python train_colab.py --mode debug  # 5 minutes, 20 samples ✅

# Options:
# --mode debug : 5 phút, test nhanh (khuyên dùng!)
# --mode quick : 15 phút, 100 samples
# --mode full  : 3 giờ, full training
```

**What changed?**

- ✅ Fixed `DataLoader worker killed` error
- ✅ Auto RAM detection (works on Colab free tier)
- ✅ Optimal settings for 12GB RAM

**Details:** See [MEMORY_FIX.md](MEMORY_FIX.md)

---

## 🌐 Google Colab (Khuyên dùng - không cần setup)

Chạy ngay trên trình duyệt với GPU miễn phí:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoangNguyennnnnnn/WaveMeshDf/blob/main/colab_quickstart.ipynb)

---

## 💻 Local Setup

### Bước 1: Cài Đặt (5 phút)

**Linux/macOS:**

```bash
git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
cd WaveMeshDf
pip install torch torchvision numpy
pip install PyWavelets trimesh matplotlib
```

**Windows:**

```cmd
git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
cd WaveMeshDf
pip install torch torchvision numpy
pip install PyWavelets trimesh matplotlib
```

**Tùy chọn - Cải thiện performance:**

```bash
pip install transformers huggingface_hub
```

Note: DINOv2 pretrained sẽ cải thiện quality, nhưng không bắt buộc

---

## Bước 2: Test Installation (2 phút)

```bash
# Test tất cả modules
python test_all_modules.py
```

**Kỳ vọng:**

```
Results: 3/4 or 4/4 modules passed
  Module A             ✅ PASS  (nếu đã cài PyWavelets)
  Module B             ✅ PASS
  Module C             ✅ PASS
  Module D             ✅ PASS
```

Nếu có lỗi, xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Bước 3: Download Data (10 phút)

### Option A: ModelNet40 (Khuyên dùng - nhanh)

```bash
python scripts/download_data.py --dataset modelnet40
```

Tự động download 500MB, extract vào `./data/ModelNet40/`

### Option B: ShapeNet (Chất lượng cao)

```bash
python scripts/download_data.py --dataset shapenet
```

Follow instructions để đăng ký tại shapenet.org, sau đó download (~50GB).

---

## Bước 4: Quick Test (5 phút)

### Test Rendering

**Linux/macOS:**

```bash
python scripts/render_multiview.py --test

python scripts/render_multiview.py \
    --mesh data/ModelNet40/chair/train/chair_0001.off \
    --output renders/
```

**Windows:**

```cmd
python scripts/render_multiview.py --test

python scripts/render_multiview.py --mesh data/ModelNet40/chair/train/chair_0001.off --output renders/
```

### Test Modules

```python
from data import mesh_to_sdf_simple, sdf_to_sparse_wavelet
import trimesh

mesh = trimesh.load('test.obj')
sdf = mesh_to_sdf_simple(mesh, resolution=32)
coeffs, coords = sdf_to_sparse_wavelet(sdf)
print(f"Sparse coefficients: {coeffs.shape}")
```

---

## Bước 5: Visualize Results (3 phút)

## Bước 5: Visualize (5 phút)

**Linux/macOS:**

```bash
python visualize_results.py
```

**Windows:**

```cmd
python visualize_results.py
```

Sẽ tạo visualization của:

- Input mesh
- SDF representation
- Sparse wavelet coefficients
- U-Net architecture

---

## Next Steps

### Training

Xem **[ROADMAP.md](ROADMAP.md)** để biết:

- Cách implement dataset loader
- Training loop hoàn chỉnh
- Evaluation metrics
- Improvement suggestions

### Documentation

- **[README.md](README.md)** - Project overview
- **[ROADMAP.md](ROADMAP.md)** - Training roadmap & improvements
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues

---

## Common Issues

### "ModuleNotFoundError: No module named 'pywt'"

```bash
pip install PyWavelets
```

### "transformers not available"

```bash
pip install transformers huggingface_hub
```

Code sẽ tự động dùng fallback CNN nếu không có transformers.

### "CUDA out of memory"

Giảm batch size hoặc resolution trong training script.

### "Rendering fails on headless server"

**Linux only:**

```bash
export PYOPENGL_PLATFORM=osmesa
pip install osmesa
```

Xem đầy đủ tại [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

**Bắt đầu ngay! 🚀**

```bash
pip install PyWavelets torch trimesh
python test_all_modules.py
python scripts/download_data.py --dataset modelnet40
```
