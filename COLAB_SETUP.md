# Google Colab Setup Guide

## 🚀 Quick Start Checklist

### 1️⃣ Enable GPU (CRITICAL for speed!)

**Why?** GPU makes training **10-50x faster** than CPU.

**Steps:**

1. Open notebook in Colab
2. Click: **Runtime → Change runtime type**
3. Select: **Hardware accelerator → T4 GPU** (free tier)
   - Or **L4 GPU** if you have Colab Pro
4. Click: **Save**
5. Wait for runtime to restart

**Verify GPU is enabled:**

```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # e.g., "Tesla T4"
```

---

### 2️⃣ Understand RAM Limits

| Colab Tier | RAM    | GPU VRAM         | Best Resolution |
| ---------- | ------ | ---------------- | --------------- |
| **Free**   | ~12 GB | ~15 GB (T4)      | 32³             |
| **Pro**    | ~25 GB | ~15 GB (T4/V100) | 64³             |
| **Pro+**   | ~50 GB | ~40 GB (A100)    | 128³            |

**Auto-detection in notebook:**
The notebook automatically detects your RAM and chooses safe resolution:

- **20+ GB RAM** → Use 64³ (high quality)
- **10-20 GB RAM** → Use 32³ (good balance)
- **<10 GB RAM** → Use 16³ (fast, lower quality)

---

### 3️⃣ Fix RAM Crashes

#### Symptoms:

- Error: "Runtime crashed"
- Error: "Ran out of memory"
- Error: "Đã xảy ra lỗi phiên sau khi bạn sử dụng toàn bộ dung lượng RAM còn trống"

#### Solutions (in order):

**A. Clear Memory:**

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

**B. Restart Runtime:**

1. Click: **Runtime → Restart runtime**
2. Run setup cells again
3. Continue from where you left off

**C. Use Lower Resolution:**

```python
# Change from:
sdf = mesh_to_sdf_simple(mesh, resolution=64)

# To:
sdf = mesh_to_sdf_simple(mesh, resolution=32)  # 8x less memory!
```

**D. Upgrade Colab (if needed):**

- Free tier: 12 GB RAM (sufficient for resolution=32)
- Colab Pro: $10/month, 25 GB RAM (can use resolution=64)

---

### 4️⃣ GPU vs CPU Performance

**Example benchmark (ModelNet40 training):**

| Device       | Resolution | Batch Size | Time/Epoch | Speed          |
| ------------ | ---------- | ---------- | ---------- | -------------- |
| **CPU**      | 32³        | 2          | ~45 min    | 1x (baseline)  |
| **GPU T4**   | 32³        | 8          | ~3 min     | **15x faster** |
| **GPU T4**   | 64³        | 4          | ~8 min     | **5x faster**  |
| **GPU A100** | 128³       | 16         | ~2 min     | **22x faster** |

**Recommendation:** Always use GPU! Even free tier T4 is 15x faster than CPU.

---

## 🔧 Common Issues & Fixes

### Issue 1: "CUDA out of memory"

**Cause:** GPU VRAM full (different from system RAM!)

**Solutions:**

1. Reduce batch size:

   ```python
   # In train.py or notebook
   batch_size = 4  # Instead of 8
   ```

2. Use gradient checkpointing (saves VRAM):

   ```python
   # In training loop
   torch.utils.checkpoint.checkpoint(model, x)
   ```

3. Use mixed precision (FP16):

   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       output = model(x)
   ```

---

### Issue 2: "Runtime disconnected"

**Cause:** Colab disconnects after ~90 minutes of inactivity (free tier)

**Solutions:**

1. Keep browser tab active
2. Use this script to keep alive:

   ```javascript
   // Run in browser console (F12)
   setInterval(() => {
     document.querySelector("colab-connect-button").click();
   }, 60000); // Click every 60 seconds
   ```

3. Save checkpoints frequently:
   ```python
   # In training loop (every 10 epochs)
   if epoch % 10 == 0:
       torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
   ```

---

### Issue 3: Code runs but resolution is still 64

**Cause:** You're running old code from notebook cell, not the updated version

**Fix:**

1. **Reload the notebook:**

   - File → Save
   - File → Close
   - Re-open notebook
   - Run cells from top

2. **Check the actual code:**
   Look for this line in the ModelNet40 cell:

   ```python
   resolution = 32  # Should be 32, not 64!
   ```

3. **Force refresh:**
   - Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)

---

### Issue 4: Slow training even with GPU

**Possible causes:**

1. **GPU not actually being used:**

   ```python
   # Check if tensors are on GPU
   print(x.device)  # Should show: cuda:0

   # Move to GPU if needed:
   x = x.to('cuda')
   model = model.to('cuda')
   ```

2. **Data transfer bottleneck:**

   ```python
   # Pin memory for faster CPU→GPU transfer
   dataloader = DataLoader(dataset, pin_memory=True)
   ```

3. **Small batch size:**
   - GPU works best with larger batches
   - Try batch_size=8 or 16 instead of 2-4

---

## 📊 Memory Monitoring

### Check RAM usage:

```python
import psutil

mem = psutil.virtual_memory()
print(f"RAM: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)")
```

### Check GPU usage:

```python
import torch

if torch.cuda.is_available():
    # Memory
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU allocated: {allocated:.1f}GB")
    print(f"GPU reserved: {reserved:.1f}GB")

    # Utilization (requires nvidia-smi)
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
```

### Monitor during training:

```python
# Add to training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # ... training code ...

        if batch_idx % 100 == 0:
            mem = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"  RAM: {mem.percent}%")
            print(f"  GPU: {gpu_mem:.1f}GB")
```

---

## 🎯 Recommended Workflow

### For First-Time Users (Colab Free):

1. **Enable GPU** (Runtime → Change runtime type → T4 GPU)

2. **Run notebook cells in order:**

   - Setup cells (install dependencies)
   - GPU check cell ← **Important!**
   - Quick demos (Module A, B, C, D)
   - ModelNet40 demo (auto-detects resolution)
   - Training demo (5 samples, 5 epochs)

3. **Expected timeline:**
   - Setup: 2-3 minutes
   - Demos: 5-10 minutes
   - Training demo: 2-5 minutes
   - **Total: ~15 minutes**

### For Full Training (Colab Free):

1. **Download ModelNet40** (~500 MB, 5-10 min)
2. **Debug training** (20 samples, 5 epochs, ~5 min)
3. **Full training** (9,843 samples, 50 epochs, ~2-3 hours)
4. **Generate meshes** (10 samples, ~2 min)

**Tips:**

- Save checkpoints every 10 epochs
- Use batch_size=8 for T4 GPU
- Use resolution=32 (safe for free tier)

### For High-Quality Results (Colab Pro):

1. **Enable L4 or V100 GPU**
2. **Use resolution=64**
3. **Use batch_size=16**
4. **Train for 100+ epochs**
5. **Enable attention layers**

---

## 💾 Data Persistence

**Problem:** Colab deletes all files when runtime disconnects!

**Solutions:**

### Option 1: Google Drive (Recommended)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
output_dir = '/content/drive/MyDrive/WaveMeshDiff/outputs'
torch.save(model.state_dict(), f'{output_dir}/checkpoint.pth')

# Load data from Drive
data_dir = '/content/drive/MyDrive/WaveMeshDiff/data'
```

### Option 2: Download Files

```python
# Download checkpoint
from google.colab import files
files.download('outputs/best.pth')

# Download generated meshes
!zip -r generated.zip generated_meshes/
files.download('generated.zip')
```

### Option 3: Git Commit (for code changes)

```python
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"
!git add .
!git commit -m "Training checkpoint"
!git push
```

---

## 🚀 Performance Tips

### 1. Use Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefit:** 2x faster, 50% less GPU memory

### 2. Increase Batch Size on GPU

```python
# CPU: batch_size=2
# T4 GPU: batch_size=8
# A100 GPU: batch_size=32
```

**Benefit:** Better GPU utilization

### 3. Use DataLoader with Multiple Workers

```python
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=2,  # Parallel data loading
    pin_memory=True  # Faster CPU→GPU transfer
)
```

**Benefit:** Reduce data loading bottleneck

### 4. Gradient Accumulation (for large models)

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit:** Simulate larger batch size without more memory

---

## 📚 Additional Resources

- **Colab Guide:** https://colab.research.google.com/notebooks/intro.ipynb
- **GPU Selection:** https://research.google.com/colaboratory/faq.html#gpu-availability
- **PyTorch GPU Tutorial:** https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#cuda-tensors

---

**Quick Reference Card:**

| Action          | Command                                                         |
| --------------- | --------------------------------------------------------------- |
| Enable GPU      | Runtime → Change runtime type → T4 GPU                          |
| Check GPU       | `torch.cuda.is_available()`                                     |
| Clear RAM       | `gc.collect()`                                                  |
| Clear GPU       | `torch.cuda.empty_cache()`                                      |
| Restart Runtime | Runtime → Restart runtime                                       |
| Check RAM       | `psutil.virtual_memory()`                                       |
| Check GPU RAM   | `torch.cuda.memory_allocated()`                                 |
| Download file   | `from google.colab import files; files.download('file.pth')`    |
| Mount Drive     | `from google.colab import drive; drive.mount('/content/drive')` |

---

**Emergency Troubleshooting:**

1. **Everything is broken** → Runtime → Restart runtime
2. **Still broken** → Runtime → Factory reset runtime
3. **Still broken** → File → Revision history → Restore older version
4. **Still broken** → Re-clone repository from GitHub

Good luck! 🚀
