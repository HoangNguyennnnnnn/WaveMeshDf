# 🗺️ Lộ Trình Tiếp Theo - WaveMesh-Diff

## 📍 Vị Trí Hiện Tại

✅ **Đã hoàn thành:**

- Module A: Wavelet Transform 3D (cần cài PyWavelets)
- Module B: Sparse U-Net (✅ tested)
- Module C: Gaussian Diffusion (✅ tested)
- Module D: Multi-view Encoder (✅ tested)
- Integration testing (✅ passed)

⚠️ **Thiếu:**

- Dependencies đầy đủ (PyWavelets, transformers, spconv)
- Dataset cho training
- Training pipeline
- Evaluation metrics

---

## 🎯 Các Bước Tiếp Theo

### **BƯỚC 1: Hoàn thiện Dependencies** ⭐ (Ưu tiên cao)

```bash
# 1.1 Cài đặt PyWavelets cho Module A
pip install PyWavelets

# 1.2 Cài transformers cho DINOv2
pip install transformers huggingface_hub
huggingface-cli login  # Cần tạo account HuggingFace (free)

# 1.3 Cài spconv cho GPU acceleration (tùy chọn, cần CUDA)
pip install spconv-cu118  # Thay cu118 bằng CUDA version của bạn
# Hoặc chạy dense mode (chậm hơn nhưng không cần GPU)

# 1.4 Dependencies khác
pip install trimesh mcubes scikit-image matplotlib tqdm
```

**Kiểm tra:**

```bash
python test_all_modules.py  # Phải pass 4/4 modules
```

---

### **BƯỚC 2: Tìm và Chuẩn Bị Dataset** 🗃️

#### **2.1 Datasets Khuyên Dùng**

##### **ShapeNet (Khuyên dùng - Phổ biến nhất)**

- **URL:** https://shapenet.org/
- **Nội dung:** 51,300 3D models, 55 categories
- **Format:** OBJ, PLY mesh files
- **Kích thước:** ~50GB
- **Ưu điểm:**
  - Dataset chuẩn cho 3D research
  - Có sẵn train/val/test split
  - Đa dạng categories (chair, car, airplane...)
  - Miễn phí (cần đăng ký)

**Download:**

```bash
# Đăng ký tài khoản tại https://shapenet.org/
# Sau khi approve, download ShapeNetCore.v2
wget https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
```

##### **ABC Dataset (Advanced - Phức tạp hơn)**

- **URL:** https://deep-geometry.github.io/abc-dataset/
- **Nội dung:** 1M+ CAD models
- **Format:** STEP, OBJ
- **Kích thước:** ~250GB
- **Ưu điểm:** Rất detailed, industrial CAD models

##### **ModelNet40 (Nhỏ hơn - Để test nhanh)**

- **URL:** https://modelnet.cs.princeton.edu/
- **Nội dung:** 12,311 models, 40 categories
- **Kích thước:** ~500MB
- **Ưu điểm:**
  - Nhẹ, download nhanh
  - Tốt cho prototyping
  - Có sẵn alignment

**Download ModelNet40 (nhanh nhất):**

```bash
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
```

##### **Objaverse (Mới nhất - Rất lớn)**

- **URL:** https://objaverse.allenai.org/
- **Nội dung:** 800K+ 3D objects
- **Format:** GLB, GLTF
- **Kích thước:** ~5TB (có thể download subset)

#### **2.2 Multi-view Images Dataset**

Vì Module D cần multi-view images, có 2 cách:

**Cách 1: Sử dụng dataset có sẵn multi-view**

- **CO3Dv2** (Facebook): https://github.com/facebookresearch/co3d
  - 19K videos, 50+ categories
  - Có camera poses
  - ~5TB

**Cách 2: Tự render từ 3D mesh** ⭐ (Khuyên dùng)

```python
# Script tự động render multi-view từ mesh
import trimesh
import numpy as np
import pyrender

def render_multiview(mesh_path, num_views=8):
    """Render mesh từ nhiều góc nhìn"""
    mesh = trimesh.load(mesh_path)

    # Setup camera positions (orbit around object)
    radius = 2.0
    images = []
    poses = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        camera_pos = [
            radius * np.cos(angle),
            radius * np.sin(angle),
            1.0  # Height
        ]

        # Render image + get pose
        img, pose = render_view(mesh, camera_pos)
        images.append(img)
        poses.append(pose)

    return images, poses
```

---

### **BƯỚC 3: Tạo Data Pipeline** 🔄

Tạo file `data/dataset.py`:

```python
import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np
from pathlib import Path

class ShapeNetDataset(Dataset):
    """
    Dataset cho ShapeNet với multi-view rendering
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 32,
        num_views: int = 4,
        category: str = None  # None = all categories
    ):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.num_views = num_views

        # Load file list
        self.mesh_files = self._load_file_list(split, category)

        print(f"Loaded {len(self.mesh_files)} meshes for {split}")

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]

        # 1. Load mesh
        mesh = trimesh.load(mesh_path)

        # 2. Convert to SDF
        from data import mesh_to_sdf_simple, sdf_to_sparse_wavelet
        sdf = mesh_to_sdf_simple(mesh, resolution=self.resolution)

        # 3. Wavelet transform
        coeffs, coords = sdf_to_sparse_wavelet(sdf)

        # 4. Render multi-view images
        images, poses = self._render_multiview(mesh)

        return {
            'coeffs': coeffs,
            'coords': coords,
            'images': images,      # (num_views, 3, 224, 224)
            'poses': poses,        # (num_views, 3, 4)
            'mesh_path': str(mesh_path)
        }

    def _render_multiview(self, mesh):
        """Render mesh từ nhiều góc"""
        # Implementation với pyrender
        # ... (chi tiết ở dưới)
        pass
```

---

### **BƯỚC 4: Tạo Training Script** 🏋️

Tạo file `train.py`:

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb  # Optional: tracking experiments

from data.dataset import ShapeNetDataset
from models import WaveMeshUNet, GaussianDiffusion, MultiViewEncoder

def train_wavemesh_diff(
    data_root: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = 'cuda'
):
    """
    Training pipeline cho WaveMesh-Diff
    """

    # 1. Setup dataset
    print("Loading dataset...")
    train_dataset = ShapeNetDataset(
        root_dir=data_root,
        split='train',
        resolution=32,
        num_views=4
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize models
    print("Initializing models...")

    # Module D: Multi-view encoder
    multiview_encoder = MultiViewEncoder(
        image_size=224,
        feature_dim=768,
        freeze_vision=True  # Freeze DINOv2
    ).to(device)

    # Module B: U-Net
    unet = WaveMeshUNet(
        in_channels=1,
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=[256, 128, 64, 32],
        time_emb_dim=256,
        use_attention=True,
        context_dim=768  # Match multiview_encoder output
    ).to(device)

    # Module C: Diffusion
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_schedule='linear'
    )

    # 3. Optimizer
    optimizer = AdamW([
        {'params': multiview_encoder.parameters(), 'lr': lr * 0.1},  # Lower LR for pretrained
        {'params': unet.parameters(), 'lr': lr}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 4. Training loop
    print("Start training...")
    for epoch in range(num_epochs):
        multiview_encoder.train()
        unet.train()

        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Move to device
            images = batch['images'].to(device)      # (B, N_views, 3, 224, 224)
            poses = batch['poses'].to(device)        # (B, N_views, 3, 4)
            coeffs = batch['coeffs'].to(device)      # (B, N_coeffs, 1)
            coords = batch['coords'].to(device)      # (B, N_coeffs, 4)

            # Forward
            # Step 1: Encode conditioning
            with torch.no_grad() if multiview_encoder.vision_encoder.freeze else torch.enable_grad():
                conditioning = multiview_encoder(images, poses)  # (B, N_views, 768)
                # Pool over views
                conditioning = conditioning.mean(dim=1)  # (B, 768)

            # Step 2: Create sparse tensor
            from models import create_sparse_tensor_from_wavelet
            x_sparse = create_sparse_tensor_from_wavelet(coeffs, coords, (32, 32, 32))

            # Step 3: Sample random timestep
            t = torch.randint(0, diffusion.timesteps, (len(images),), device=device)

            # Step 4: Diffusion loss
            # Add noise to x_sparse
            noise = torch.randn_like(coeffs)
            x_noisy = diffusion.q_sample(coeffs, t, noise)

            # Predict noise
            x_noisy_sparse = create_sparse_tensor_from_wavelet(x_noisy, coords, (32, 32, 32))
            pred_noise = unet(x_noisy_sparse, t, context=conditioning)

            # Loss
            loss = torch.nn.functional.mse_loss(pred_noise.features, noise)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Scheduler step
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'unet': unet.state_dict(),
                'multiview_encoder': multiview_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")

        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")

if __name__ == '__main__':
    train_wavemesh_diff(
        data_root='./data/ShapeNetCore.v2',
        output_dir='./checkpoints',
        num_epochs=100,
        batch_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
```

---

### **BƯỚC 5: Cải Tiến Code** 🚀

#### **5.1 Tối Ưu Performance**

**Cải tiến 1: Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    loss = diffusion(x_sparse, context=conditioning)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Cải tiến 2: Gradient Accumulation**

```python
# Cho batch size lớn hơn khi GPU memory hạn chế
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Cải tiến 3: EMA (Exponential Moving Average)**

```python
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(unet.parameters(), decay=0.9999)

# After optimizer.step()
ema.update()

# For inference
with ema.average_parameters():
    output = unet(x, t)
```

#### **5.2 Cải Tiến Architecture**

**Cải tiến 1: Adaptive Layer Norm (AdaLN)**

```python
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.time_mlp = nn.Linear(time_dim, dim * 2)

    def forward(self, x, time_emb):
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=-1)
        return self.ln(x) * (1 + scale) + shift
```

**Cải tiến 2: Classifier-Free Guidance**

```python
def train_with_cfg(self, x, t, context, drop_prob=0.1):
    # Random drop conditioning
    mask = torch.rand(len(x)) > drop_prob
    context = context * mask.unsqueeze(-1)

    return self.unet(x, t, context)

def sample_with_cfg(self, shape, context, guidance_scale=7.5):
    # Sample with và without conditioning
    noise_pred_cond = self.unet(x_t, t, context)
    noise_pred_uncond = self.unet(x_t, t, None)

    # Guided prediction
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return noise_pred
```

**Cải tiến 3: Multi-scale Features**

```python
class MultiScaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Extract features ở nhiều resolutions
        self.scales = [224, 112, 56]

    def forward(self, images):
        features = []
        for scale in self.scales:
            img_resized = F.interpolate(images, size=scale)
            feat = self.encoder(img_resized)
            features.append(feat)
        return torch.cat(features, dim=-1)
```

#### **5.3 Thêm Evaluation Metrics**

```python
def evaluate_model(model, test_loader, device):
    """
    Đánh giá chất lượng 3D mesh generation
    """
    from scipy.spatial.distance import directed_hausdorff

    metrics = {
        'chamfer_distance': [],
        'f_score': [],
        'iou': []
    }

    for batch in test_loader:
        # Generate samples
        generated = model.sample(...)
        ground_truth = batch['mesh']

        # Chamfer Distance
        cd = chamfer_distance(generated, ground_truth)
        metrics['chamfer_distance'].append(cd)

        # F-Score (precision/recall at threshold)
        f1 = f_score(generated, ground_truth, threshold=0.01)
        metrics['f_score'].append(f1)

        # IoU
        iou = compute_iou(generated, ground_truth)
        metrics['iou'].append(iou)

    return {k: np.mean(v) for k, v in metrics.items()}
```

---

### **BƯỚC 6: Visualization & Debugging** 📊

Tạo file `visualize_training.py`:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_generation(images, generated_mesh, ground_truth_mesh):
    """
    Visualize input images + generated mesh + GT mesh
    """
    fig = plt.figure(figsize=(15, 5))

    # Plot input views
    for i in range(4):
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(images[i].permute(1, 2, 0).cpu())
        ax.axis('off')
        ax.set_title(f'View {i+1}')

    # Plot generated mesh
    ax = fig.add_subplot(2, 4, 5, projection='3d')
    plot_mesh_3d(ax, generated_mesh)
    ax.set_title('Generated')

    # Plot ground truth
    ax = fig.add_subplot(2, 4, 6, projection='3d')
    plot_mesh_3d(ax, ground_truth_mesh)
    ax.set_title('Ground Truth')

    plt.tight_layout()
    return fig
```

---

## 📋 Checklist Thực Hiện

### **Tuần 1: Setup & Data**

- [ ] Cài đặt tất cả dependencies
- [ ] Download ModelNet40 (test nhanh)
- [ ] Implement rendering script
- [ ] Test data pipeline với 1 sample

### **Tuần 2: Training Infrastructure**

- [ ] Implement `dataset.py`
- [ ] Implement `train.py`
- [ ] Test training với 10 samples (overfitting test)
- [ ] Setup wandb/tensorboard logging

### **Tuần 3-4: Full Training**

- [ ] Download ShapeNet (hoặc subset)
- [ ] Train trên single category (e.g., chairs)
- [ ] Implement evaluation metrics
- [ ] Visualize results

### **Tuần 5+: Optimization**

- [ ] Train trên full dataset
- [ ] Hyperparameter tuning
- [ ] Implement improvements (CFG, EMA, etc.)
- [ ] Write paper/report

---

## 🎓 Tài Nguyên Học Tập

### **Papers Nên Đọc:**

1. **Diffusion Models:**
   - DDPM (Denoising Diffusion Probabilistic Models)
   - DDIM (Faster sampling)
2. **3D Generation:**

   - Point-E (OpenAI)
   - Shap-E (OpenAI)
   - DreamFusion (Text-to-3D)

3. **Multi-view 3D:**
   - NeRF (Neural Radiance Fields)
   - Zero-1-to-3 (Single image to 3D)

### **Code References:**

- https://github.com/CompVis/stable-diffusion (Diffusion training)
- https://github.com/lucidrains/denoising-diffusion-pytorch (Clean implementation)
- https://github.com/openai/point-e (3D diffusion)

### **Communities:**

- Papers With Code: https://paperswithcode.com/
- Hugging Face Forums: https://discuss.huggingface.co/
- Reddit r/MachineLearning

---

## 💡 Tips & Best Practices

1. **Start Small:**

   - Train trên 1 category trước (e.g., chairs)
   - Dùng resolution thấp (16x16x16) để test nhanh
   - Overfit trên 10 samples để verify code đúng

2. **Monitor Training:**

   - Log loss curves
   - Visualize generations mỗi epoch
   - Track GPU memory usage

3. **Save Everything:**

   - Checkpoints mỗi epoch
   - Logs và configs
   - Generated samples

4. **Incremental Development:**
   - Implement baseline trước
   - Add improvements từng cái một
   - A/B test mỗi thay đổi

---

## 🚨 Common Issues & Solutions

**Issue 1: Out of Memory**

```python
# Solution: Reduce batch size, use gradient accumulation
batch_size = 4  # Thay vì 16
accumulation_steps = 4
```

**Issue 2: Training Diverges**

```python
# Solution: Lower learning rate, clip gradients
lr = 1e-5  # Thay vì 1e-4
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Issue 3: Poor Generation Quality**

```python
# Solution: More timesteps, better conditioning
timesteps = 2000  # Thay vì 1000
guidance_scale = 7.5  # Classifier-free guidance
```

---

## 📞 Khi Cần Giúp Đỡ

1. Check error logs carefully
2. Search GitHub Issues của similar projects
3. Ask on forums (Stack Overflow, Hugging Face)
4. Document your experiments

**Good luck! 🚀**
