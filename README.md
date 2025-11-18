# WaveMesh-Diff

**3D Mesh Generation using Diffusion Models in Wavelet Domain**

Phát sinh 3D mesh từ multi-view images sử dụng diffusion models trên sparse 3D wavelet coefficients.

---

## 🎯 Tổng Quan

WaveMesh-Diff kết hợp 4 modules chính:

1. **Module A - Wavelet Transform**: Chuyển 3D SDF → sparse wavelet coefficients
2. **Module B - Sparse U-Net**: Denoising network cho diffusion
3. **Module C - Gaussian Diffusion**: DDPM/DDIM training và sampling
4. **Module D - Multi-view Encoder**: Encode images từ nhiều góc nhìn

**Ưu điểm:**
- ✅ Tiết kiệm memory (sparse representation)
- ✅ Scalable (có thể tăng resolution)
- ✅ Conditioning từ multi-view images
- ✅ Topology-consistent meshing

---

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt

```bash
# Clone repository
git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
cd WaveMeshDf

# Cài dependencies cơ bản
pip install torch torchvision numpy
pip install PyWavelets trimesh matplotlib

# Tùy chọn: Cài đầy đủ
pip install transformers huggingface_hub  # Cho DINOv2
pip install pyrender                      # Cho rendering
```

### 2. Test Installation

```bash
# Test tất cả modules
python test_all_modules.py

# Kỳ vọng: 3/4 hoặc 4/4 modules PASS
# (Module A cần PyWavelets)
```

### 3. Quick Example

```python
# Example: Sử dụng các modules
from data import mesh_to_sdf_simple, sdf_to_sparse_wavelet
from models import WaveMeshUNet, GaussianDiffusion, MultiViewEncoder

# Module A: Mesh → Wavelet
sdf = mesh_to_sdf_simple(mesh, resolution=32)
coeffs, coords = sdf_to_sparse_wavelet(sdf)

# Module D: Multi-view encoding
encoder = MultiViewEncoder(feature_dim=768)
conditioning = encoder(images, camera_poses)

# Module B + C: Diffusion
unet = WaveMeshUNet(context_dim=768, use_attention=True)
diffusion = GaussianDiffusion()

# Training
loss = diffusion(x, context=conditioning)
```

**📖 Xem [QUICKSTART.md](QUICKSTART.md) để biết chi tiết.**

---

## 📁 Cấu Trúc Project

```
WaveMesh-Diff/
├── data/
│   ├── __init__.py
│   └── wavelet_utils.py          # Module A: Wavelet transform
├── models/
│   ├── __init__.py
│   ├── unet_sparse.py            # Module B: Sparse U-Net
│   ├── diffusion.py              # Module C: Diffusion model
│   ├── multiview_encoder.py      # Module D: Multi-view encoder
│   └── spconv_compat.py          # Sparse conv compatibility layer
├── scripts/
│   ├── download_data.py          # Download datasets
│   └── render_multiview.py       # Render multi-view images
├── tests/
│   ├── test_spconv_compat.py
│   ├── test_modules_bc.py
│   └── test_wavelet_pipeline.py
├── test_all_modules.py           # Comprehensive testing
├── test_module_d.py              # Module D testing
├── visualize_results.py          # Visualization script
├── requirements.txt
├── README.md                     # This file
├── QUICKSTART.md                 # Quick start guide
├── ROADMAP.md                    # Development roadmap
├── ARCHITECTURE.md               # Technical details
└── TROUBLESHOOTING.md            # Common issues
```

---

## 🏗️ Architecture Overview

### Module A: Wavelet Transform 3D

Chuyển đổi giữa 3D SDF và sparse wavelet coefficients.

**API:**
```python
from data import mesh_to_sdf_simple, sdf_to_sparse_wavelet, sparse_wavelet_to_sdf

# Mesh → SDF → Wavelet
sdf = mesh_to_sdf_simple(mesh, resolution=32)
coeffs, coords = sdf_to_sparse_wavelet(sdf, threshold=0.01)

# Reconstruct
sdf_recon = sparse_wavelet_to_sdf(coeffs, coords, shape=(32,32,32))
```

### Module B: Sparse U-Net

3D U-Net với sparse convolutions, time embedding, và cross-attention.

**API:**
```python
from models import WaveMeshUNet

model = WaveMeshUNet(
    in_channels=1,
    encoder_channels=[32, 64, 128],
    decoder_channels=[128, 64, 32],
    time_emb_dim=256,
    use_attention=True,
    context_dim=768  # Cho conditioning
)

output = model(x_sparse, timestep, context=conditioning)
```

### Module C: Gaussian Diffusion

DDPM và DDIM diffusion process.

**API:**
```python
from models import GaussianDiffusion

diffusion = GaussianDiffusion(
    timesteps=1000,
    beta_schedule='linear'
)

# Training
loss = diffusion.compute_loss(x_start)

# Sampling
samples = diffusion.sample(shape=(B, C, H, W, D), method='ddim', steps=50)
```

### Module D: Multi-view Encoder

Encode multi-view images thành conditioning features.

**API:**
```python
from models import MultiViewEncoder, create_multiview_encoder

# Cách 1: Manual
encoder = MultiViewEncoder(
    image_size=224,
    feature_dim=768,
    num_heads=8
)

# Cách 2: Preset
encoder = create_multiview_encoder(preset='base')  # 'small', 'base', 'large'

# Usage
images = torch.randn(B, N_views, 3, 224, 224)
poses = torch.randn(B, N_views, 3, 4)
conditioning = encoder(images, poses)  # (B, N_views, 768)
```

**📖 Xem [ARCHITECTURE.md](ARCHITECTURE.md) để biết chi tiết kỹ thuật.**

---

## 📊 Training

### Chuẩn Bị Data

```bash
# Download ModelNet40 (500MB - quick start)
python scripts/download_data.py --dataset modelnet40

# Hoặc download ShapeNet (50GB - better quality)
python scripts/download_data.py --dataset shapenet
# Follow instructions để đăng ký
```

### Training Pipeline

Xem **[ROADMAP.md](ROADMAP.md)** để có:
- Dataset implementation đầy đủ
- Training loop với all 4 modules
- Evaluation metrics
- Improvement suggestions

### Quick Test

```bash
# Overfit test (verify code works)
python train_simple.py --num_samples 10 --num_epochs 50

# Kỳ vọng: Loss từ ~0.5 → ~0.01
```

---

## 🧪 Testing

```bash
# Test tất cả modules
python test_all_modules.py

# Test riêng Module D
python test_module_d.py

# Test specific modules
python -m pytest tests/ -v
```

**Test Results:**
- ✅ Module B: Sparse U-Net (395K params)
- ✅ Module C: Gaussian Diffusion (DDPM/DDIM)
- ✅ Module D: Multi-view Encoder (with fallback)
- ⚠️ Module A: Cần cài PyWavelets

---

## 📈 Performance

### Current Status

- **Backend**: Dense fallback mode (chưa cài spconv)
- **Vision**: CNN fallback (chưa cài transformers)
- **Status**: ✅ All modules tested và hoạt động

### Production Setup

```bash
# Full performance
pip install spconv-cu118          # GPU sparse convolutions
pip install transformers          # Pre-trained DINOv2
huggingface-cli login            # Download DINOv2 weights
```

### Expected Speed

| Setup | Resolution | Time/Epoch | Hardware |
|-------|-----------|-----------|----------|
| CPU Dense | 32³ | ~30 min | i7 |
| GPU Dense | 32³ | ~5 min | RTX 3080 |
| GPU Sparse | 32³ | ~2 min | RTX 3080 + spconv |
| GPU Sparse | 64³ | ~8 min | RTX 3080 + spconv |

---

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'pywt'"**
```bash
pip install PyWavelets
```

**"transformers not available"**
```bash
pip install transformers huggingface_hub
# Code tự động fallback sang CNN
```

**"CUDA out of memory"**
```bash
# Giảm batch size hoặc resolution
python train.py --batch_size 2 --resolution 16
```

**📖 Xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md) để biết thêm chi tiết.**

---

## 📚 Documentation

- **[README.md](README.md)** - Project overview (file này)
- **[QUICKSTART.md](QUICKSTART.md)** - Bắt đầu trong 30 phút
- **[ROADMAP.md](ROADMAP.md)** - Lộ trình training & improvement
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Chi tiết kỹ thuật
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Giải quyết lỗi

---

## 🎯 Roadmap

### Hiện Tại (v0.1)
- ✅ 4 modules hoàn chỉnh
- ✅ Testing infrastructure
- ✅ Documentation
- ⚠️ Chưa có trained weights

### Tiếp Theo (v0.2)
- [ ] Training scripts hoàn chỉnh
- [ ] Pre-trained weights
- [ ] Evaluation metrics
- [ ] Demo notebooks

### Tương Lai (v1.0)
- [ ] Multi-GPU training
- [ ] Classifier-free guidance
- [ ] Progressive training
- [ ] Web demo

**📖 Xem [ROADMAP.md](ROADMAP.md) để biết chi tiết.**

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- **Diffusion Models**: DDPM, DDIM papers
- **3D Generation**: Point-E, Shap-E (OpenAI)
- **Vision Encoder**: DINOv2 (Meta)
- **Datasets**: ShapeNet, ModelNet40

---

## 📞 Contact

- **GitHub**: [HoangNguyennnnnnn/WaveMeshDf](https://github.com/HoangNguyennnnnnn/WaveMeshDf)
- **Issues**: [Report bugs](https://github.com/HoangNguyennnnnnn/WaveMeshDf/issues)

---

**Happy 3D Generation! 🎨**
