# 📘 WaveMesh-Diff - Giải Thích Chi Tiết

## 🎯 Chúng Ta Đang Làm Gì?

**Mục tiêu chính:** Tạo ra 3D mesh (mô hình 3D) từ hình ảnh bằng AI

**Cách làm:** Sử dụng Diffusion Model (như Stable Diffusion nhưng cho 3D)

---

## 🏗️ Kiến Trúc Tổng Thể

Project được chia thành **4 modules chính**:

```
┌─────────────────────────────────────────────────────────────┐
│                    WAVEMESH-DIFF PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📸 INPUT: Multi-view Images (4-6 ảnh từ các góc khác nhau) │
│                           │                                  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────┐      │
│  │ MODULE D: Multi-view Encoder (TODO - Chưa làm)   │      │
│  │ - Sử dụng DINOv2 để encode ảnh thành features    │      │
│  │ - Output: Image embeddings (768-dim)              │      │
│  └───────────────────────────────────────────────────┘      │
│                           │                                  │
│                           ↓ (conditioning)                   │
│  ┌───────────────────────────────────────────────────┐      │
│  │ MODULE C: Diffusion Model ✅ ĐÃ XONG              │      │
│  │ - Gaussian Diffusion (DDPM/DDIM)                  │      │
│  │ - Thêm noise vào dữ liệu rồi học cách denoise     │      │
│  │ - Output: Clean wavelet coefficients              │      │
│  └───────────────────────────────────────────────────┘      │
│                           │                                  │
│                           ↓ (uses at each denoising step)   │
│  ┌───────────────────────────────────────────────────┐      │
│  │ MODULE B: Sparse 3D U-Net ✅ ĐÃ XONG              │      │
│  │ - Neural network để denoise wavelet coefficients  │      │
│  │ - Encoder-Decoder architecture                    │      │
│  │ - Hoạt động trên sparse data (tiết kiệm memory)   │      │
│  └───────────────────────────────────────────────────┘      │
│                           │                                  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────┐      │
│  │ MODULE A: Wavelet Transform ✅ ĐÃ XONG            │      │
│  │ - Chuyển đổi giữa Mesh ↔ SDF ↔ Wavelet           │      │
│  │ - Nén dữ liệu 50-500x (từ 64MB xuống 1.3MB)     │      │
│  │ - Output: 3D Mesh (.obj file)                     │      │
│  └───────────────────────────────────────────────────┘      │
│                           │                                  │
│                           ↓                                  │
│  🎨 OUTPUT: 3D Mesh (vertices + faces)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Chi Tiết Từng Module

### MODULE A: Wavelet Transform ✅ (Hoàn thành 100%)

**Làm gì:**
Chuyển đổi 3D mesh qua lại giữa các dạng biểu diễn khác nhau.

**Pipeline:**

```
3D Mesh (.obj)
    ↓ mesh_to_sdf_simple()
Dense SDF Grid (256³ = 16 triệu voxels, ~64MB)
    ↓ sdf_to_sparse_wavelet()
Sparse Wavelet (160K coefficients, ~1.3MB) ← 50x nhỏ hơn!
    ↓ sparse_wavelet_to_sdf()
Reconstructed SDF
    ↓ sdf_to_mesh()
3D Mesh (.obj)
```

**Tại sao cần:**

- **Dense voxel grid quá lớn:** 256³ = 16 triệu voxels = 64MB cho 1 mesh
- **Wavelet transform nén xuống:** Chỉ còn ~1-2% coefficients quan trọng
- **Sparse representation:** Chỉ lưu coefficients khác 0 → tiết kiệm 98% memory

**Code example:**

```python
import trimesh
from data import (
    mesh_to_sdf_simple,
    sdf_to_sparse_wavelet,
    sparse_wavelet_to_sdf,
    sdf_to_mesh
)

# 1. Load mesh
mesh = trimesh.load("bunny.obj")

# 2. Mesh → SDF (chuyển thành grid 3D)
sdf = mesh_to_sdf_simple(mesh, resolution=64)

# 3. SDF → Sparse Wavelet (nén xuống)
sparse_data = sdf_to_sparse_wavelet(sdf, threshold=0.01)
print(f"Nén từ {64**3} xuống {len(sparse_data['features'])} coefficients")

# 4. Reconstruct lại
reconstructed_sdf = sparse_wavelet_to_sdf(sparse_data)
vertices, faces = sdf_to_mesh(reconstructed_sdf)

# 5. Save
mesh_out = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh_out.export("output.obj")
```

---

### MODULE B: Sparse 3D U-Net ✅ (Hoàn thành 100%)

**Làm gì:**
Neural network để denoise (khử nhiễu) sparse wavelet coefficients.

**Kiến trúc:**

```
Input: Noisy Wavelet Coefficients + Timestep + Context
    ↓
┌─────────────────────────┐
│ Time Embedding          │ ← Embed timestep t vào vector
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Encoder (Downsample)    │ ← [16, 32, 64] channels
│ - Sparse Conv 3D        │
│ - Residual Blocks       │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Decoder (Upsample)      │ ← [64, 32, 16] channels
│ - Sparse Transpose Conv │
│ - Skip Connections      │
│ - Cross Attention       │ ← Condition trên image features
└─────────────────────────┘
    ↓
Output: Predicted Noise (hoặc Clean Coefficients)
```

**Đặc điểm:**

- **Sparse Convolutions:** Chỉ tính toán trên voxels khác 0 → nhanh hơn 10-100x
- **Dense Fallback:** Tự động dùng PyTorch thường nếu không có spconv → chạy được trên Colab
- **Cross-Attention:** Để kết hợp thông tin từ multi-view images (cho Module D sau này)

**Code example:**

```python
from models import WaveMeshUNet, SparseConvTensor
import torch

# Create model
model = WaveMeshUNet(
    in_channels=1,
    out_channels=1,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16],
    use_attention=False
)

# Create sparse input
features = torch.randn(100, 1)  # 100 points, 1 channel
indices = torch.randint(0, 16, (100, 4))  # [batch, x, y, z]
sparse_input = SparseConvTensor(features, indices, (16,16,16), batch_size=1)

# Forward pass
timestep = torch.tensor([500])
output = model(sparse_input, timestep)
```

---

### MODULE C: Diffusion Model ✅ (Hoàn thành 100%)

**Làm gì:**
Học cách tạo ra wavelet coefficients mới từ noise thuần túy.

**Cách hoạt động (giống Stable Diffusion):**

1. **Training (Forward process):**

   ```
   Clean Data → Add Noise → Noisy Data
   x₀ → x₁ → x₂ → ... → xₜ → ... → x₁₀₀₀ (pure noise)

   Model học predict: noise hoặc x₀ từ xₜ
   ```

2. **Sampling (Reverse process):**
   ```
   Pure Noise → Denoise → ... → Clean Data
   x₁₀₀₀ → x₉₉₉ → ... → x₁ → x₀ (generated mesh!)
   ```

**Code example:**

```python
from models import GaussianDiffusion, WaveMeshUNet

# Create diffusion model
unet = WaveMeshUNet(...)
diffusion = GaussianDiffusion(
    model=unet,
    timesteps=1000,
    beta_schedule='linear'
)

# Training
loss = diffusion.training_losses(model, x_clean, t, context)
loss['mse'].backward()

# Sampling (generate new mesh)
generated = diffusion.sample(batch_size=1, context=image_features)
```

---

### MODULE D: Multi-view Encoder 🚧 (TODO - Chưa làm)

**Làm gì:**
Chuyển đổi 4-6 ảnh 2D thành features để condition diffusion model.

**Sẽ dùng:**

- DINOv2 (pre-trained vision encoder)
- Camera pose embeddings
- Cross-attention mechanism (đã chuẩn bị sẵn trong Module B)

---

## 🔧 Tại Sao Có 2 Backends (spconv vs dense)?

### 1. **spconv Backend (Optimal - Nhanh)**

- Sparse convolution chuyên dụng cho 3D data
- Nhanh hơn 10-100x so với dense
- **Vấn đề:** Cần compile C++/CUDA → không chạy được trên Colab

### 2. **Dense Fallback Backend (Colab-friendly - Chậm)**

- Dùng PyTorch thường (nn.Conv3d)
- Chậm hơn nhưng **chạy được trên Colab**
- Tự động activate khi không có spconv

**Code tự động chọn backend:**

```python
from models.spconv_compat import get_backend_info

info = get_backend_info()
print(info['backend'])  # 'spconv' hoặc 'dense_fallback'
```

---

## 📁 Cấu Trúc File

```
WaveMesh-Diff/
├── data/
│   ├── __init__.py           ← Exports các functions
│   └── wavelet_utils.py      ← MODULE A: Wavelet transform
│
├── models/
│   ├── __init__.py           ← Exports models
│   ├── spconv_compat.py      ← Compatibility layer (spconv/dense)
│   ├── unet_sparse.py        ← MODULE B: Sparse U-Net
│   └── diffusion.py          ← MODULE C: Diffusion model
│
├── tests/
│   ├── test_wavelet_pipeline.py   ← Test Module A
│   ├── test_modules_bc.py         ← Test Modules B & C
│   └── test_spconv_compat.py      ← Test compatibility layer
│
├── run_pipeline.py           ← Chạy toàn bộ pipeline
├── visualize_results.py      ← Visualize kết quả
│
├── README.md                 ← Hướng dẫn nhanh
├── SETUP_GUIDE.md            ← Hướng dẫn cài đặt
├── PIPELINE_GUIDE.md         ← Hướng dẫn sử dụng pipeline
├── VISUALIZATION_GUIDE.md    ← Hướng dẫn visualization
├── ARCHITECTURE.md           ← Kiến trúc chi tiết
├── TROUBLESHOOTING.md        ← Giải quyết lỗi
└── DOCS_INDEX.md             ← Chỉ mục toàn bộ docs
```

---

## 🚀 Cách Sử Dụng

### 1. Quick Start (Google Colab)

```python
# Cài đặt
!pip install -q PyWavelets trimesh scikit-image scipy numpy torch torchvision

# Clone repo
!git clone https://github.com/HoangNguyennnnnnn/WaveMeshDf.git
%cd WaveMeshDf

# Chạy toàn bộ pipeline
!python run_pipeline.py --resolution 32

# Visualize kết quả
!python visualize_results.py
```

### 2. Test Từng Module

**Test Module A (Wavelet):**

```bash
python tests/test_wavelet_pipeline.py --create-test-mesh --resolution 64
```

**Test Modules B & C (Neural Networks):**

```bash
python tests/test_modules_bc.py
```

### 3. API Sử Dụng

**Module A - Convenience API:**

```python
from data import (
    mesh_to_sdf_simple,      # Mesh → SDF
    sdf_to_sparse_wavelet,   # SDF → Sparse Wavelet
    sparse_wavelet_to_sdf,   # Sparse Wavelet → SDF
    sdf_to_mesh,             # SDF → Mesh
    normalize_mesh           # Normalize mesh
)
```

**Module B - Neural Network:**

```python
from models import WaveMeshUNet, SparseConvTensor

model = WaveMeshUNet(
    in_channels=1,
    out_channels=1,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16]
)
```

**Module C - Diffusion:**

```python
from models import GaussianDiffusion

diffusion = GaussianDiffusion(
    model=unet,
    timesteps=1000
)
```

---

## 🎓 Các Khái Niệm Quan Trọng

### 1. **SDF (Signed Distance Field)**

- Grid 3D, mỗi voxel lưu khoảng cách đến surface
- Giá trị âm = inside, dương = outside, 0 = trên surface
- Dùng để biểu diễn 3D shape

### 2. **Wavelet Transform**

- Giống Fourier transform nhưng tốt hơn cho signal có locality
- Tách tín hiệu thành các frequency bands
- 3D DWT: áp dụng wavelet transform theo 3 chiều

### 3. **Sparse Representation**

- Chỉ lưu các giá trị khác 0
- Format: indices + features
- Tiết kiệm 95-99% memory

### 4. **Diffusion Model**

- Học bằng cách thêm noise rồi học denoise
- Reverse process tạo ra dữ liệu mới
- State-of-the-art cho generative AI

### 5. **U-Net**

- Architecture có encoder-decoder với skip connections
- Tốt cho image-to-image tasks
- Sparse U-Net: version cho sparse 3D data

---

## 📊 Hiệu Suất

| Module   | Colab (Dense)  | Local (spconv) | Memory               |
| -------- | -------------- | -------------- | -------------------- |
| Module A | ✅ Nhanh       | ✅ Nhanh       | 1-2 MB (sparse)      |
| Module B | ⚠️ Chậm 10-50x | ✅ Nhanh       | Phụ thuộc resolution |
| Module C | ⚠️ Chậm 10-50x | ✅ Nhanh       | Phụ thuộc batch size |

**Khuyến nghị:**

- ✅ **Colab:** Tốt cho học tập, test, prototype (resolution ≤ 64)
- ⚡ **Local GPU:** Cần thiết cho training, production (resolution ≥ 128)

---

## 🐛 Lỗi Thường Gặp

### 1. `TypeError: unexpected keyword argument 'spatial_shape'`

**Nguyên nhân:** Gọi WaveMeshUNet với parameter không tồn tại  
**Giải pháp:** Đã sửa trong visualize_results.py

### 2. `spconv not available`

**Nguyên nhân:** Bình thường, Colab không có spconv  
**Giải pháp:** Không cần làm gì, tự động dùng dense fallback

### 3. Module A chậm (Converting mesh to SDF)

**Nguyên nhân:** Resolution cao (128³+) mất thời gian  
**Giải pháp:** Dùng resolution thấp hơn (32 hoặc 64) khi test

---

## 🎯 Roadmap

### ✅ Đã Hoàn Thành

- [x] Module A: Wavelet Transform
- [x] Module B: Sparse U-Net
- [x] Module C: Diffusion Model
- [x] Dense fallback cho Colab
- [x] Comprehensive documentation
- [x] Visualization tools

### 🚧 Đang Làm

- [ ] Module D: Multi-view Encoder
- [ ] Training pipeline
- [ ] Dataset loader

### 📅 Kế Hoạch Tương Lai

- [ ] End-to-end training
- [ ] Multi-GPU support
- [ ] Web demo
- [ ] Pre-trained weights

---

## 📚 Tài Liệu Tham Khảo

- **Diffusion Models:** "Denoising Diffusion Probabilistic Models" (DDPM)
- **Sparse Convolution:** spconv library
- **3D Wavelets:** PyWavelets documentation
- **SDF:** "DeepSDF: Learning Continuous Signed Distance Functions"

---

## 💡 Tóm Tắt Ngắn Gọn

**Chúng ta đang làm gì?**
→ Tạo 3D mesh từ ảnh bằng Diffusion Model

**Cách làm?**

1. Biểu diễn 3D mesh bằng sparse wavelet coefficients (Module A)
2. Train neural network denoise wavelet coefficients (Module B)
3. Sử dụng diffusion process để generate (Module C)
4. Condition trên multi-view images (Module D - TODO)

**Tại sao phức tạp?**

- 3D data rất lớn → cần sparse representation
- Diffusion model cần denoise nhiều lần → cần network nhanh
- Colab không có spconv → cần dense fallback

**Hiện tại đã làm được gì?**

- ✅ 3/4 modules hoàn thành
- ✅ Chạy được trên Colab
- ✅ Documentation đầy đủ

**Bước tiếp theo?**

1. Test các modules đã làm
2. Implement Module D
3. Training end-to-end
4. Generate 3D mesh từ ảnh!

---

**Có câu hỏi?** Check các file docs khác hoặc hỏi trực tiếp! 😊
