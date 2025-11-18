# 🚀 Quick Start - WaveMesh-Diff

Bắt đầu sử dụng WaveMesh-Diff trong 30 phút.

---

## Bước 1: Cài Đặt (5 phút)

```bash
# Clone repository
git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
cd WaveMeshDf

# Dependencies cơ bản (bắt buộc)
pip install torch torchvision numpy
pip install PyWavelets trimesh matplotlib

# Tùy chọn: Cải thiện performance
pip install transformers huggingface_hub  # DINOv2 pretrained
pip install spconv-cu118                  # GPU sparse ops (thay cu118 theo CUDA version)
```

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

```bash
# Test rendering script
python scripts/render_multiview.py --test

# Render specific mesh
python scripts/render_multiview.py \
    --mesh data/ModelNet40/chair/train/chair_0001.off \
    --output renders/
```

### Test Modules

```python
# Test Module A - Wavelet
from data import mesh_to_sdf_simple, sdf_to_sparse_wavelet
import trimesh

mesh = trimesh.load('test.obj')
sdf = mesh_to_sdf_simple(mesh, resolution=32)
coeffs, coords = sdf_to_sparse_wavelet(sdf)
print(f"Sparse coefficients: {coeffs.shape}")

# Test Module D - MultiView Encoder
from models import create_multiview_encoder
import torch

encoder = create_multiview_encoder(preset='small')
images = torch.randn(2, 4, 3, 224, 224)  # 2 batches, 4 views
poses = torch.randn(2, 4, 3, 4)
conditioning = encoder(images, poses)
print(f"Conditioning: {conditioning.shape}")  # (2, 4, 384)
```

---

## Bước 5: Visualize Results (3 phút)

```bash
# Visualize pipeline
python visualize_results.py
```

Sẽ tạo visualization của:

- Input mesh
- SDF representation
- Sparse wavelet coefficients
- U-Net architecture

---

## Next Steps

### Để Train Model:

Xem **[ROADMAP.md](ROADMAP.md)** để biết:

- Cách implement dataset loader
- Training loop hoàn chỉnh
- Evaluation metrics
- Improvement suggestions

### Code Examples:

```python
# Training example (conceptual - xem ROADMAP.md để có full code)
from data import ShapeNetDataset
from models import WaveMeshUNet, GaussianDiffusion, MultiViewEncoder
from torch.utils.data import DataLoader

# 1. Prepare data
dataset = ShapeNetDataset('data/ModelNet40', split='train')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. Initialize models
encoder = MultiViewEncoder(feature_dim=768)
unet = WaveMeshUNet(context_dim=768, use_attention=True)
diffusion = GaussianDiffusion(timesteps=1000)

# 3. Training loop
for batch in loader:
    # Encode conditioning
    conditioning = encoder(batch['images'], batch['poses'])

    # Diffusion forward
    loss = diffusion(batch['coeffs'], context=conditioning)
    loss.backward()
    optimizer.step()
```

---

## Common Issues

### "ModuleNotFoundError: No module named 'pywt'"

```bash
pip install PyWavelets
```

### "transformers not available"

```bash
pip install transformers huggingface_hub
# Hoặc code sẽ tự động dùng fallback CNN
```

### "CUDA out of memory"

```bash
# Giảm batch size hoặc resolution
python train.py --batch_size 2 --resolution 16
```

### "Rendering fails on headless server"

```bash
export PYOPENGL_PLATFORM=osmesa
pip install osmesa
```

Xem đầy đủ tại [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Performance Tips

### Tăng Tốc Training:

```bash
# 1. Cài spconv cho GPU sparse operations
pip install spconv-cu118

# 2. Sử dụng mixed precision
# Add to training: torch.cuda.amp.autocast()

# 3. Cài transformers cho pre-trained DINOv2
pip install transformers
huggingface-cli login
```

### Expected Performance:

| Setup      | Resolution | Time/Epoch | Hardware          |
| ---------- | ---------- | ---------- | ----------------- |
| CPU        | 32³        | ~30 min    | i7                |
| GPU Dense  | 32³        | ~5 min     | RTX 3080          |
| GPU Sparse | 32³        | ~2 min     | RTX 3080 + spconv |

---

## Documentation

- **[README.md](README.md)** - Project overview
- **[ROADMAP.md](ROADMAP.md)** - Training roadmap & improvements
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues

---

**Bắt đầu ngay! 🚀**

```bash
pip install PyWavelets torch trimesh
python test_all_modules.py
python scripts/download_data.py --dataset modelnet40
```
