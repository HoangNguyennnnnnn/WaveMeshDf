# 🎨 WaveMesh-Diff Visualization Guide

Complete guide for visualizing results from the WaveMesh-Diff pipeline.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Visualization Types](#visualization-types)
3. [Google Colab Usage](#google-colab-usage)
4. [Local Usage](#local-usage)
5. [Custom Visualizations](#custom-visualizations)
6. [Examples](#examples)

---

## 🚀 Quick Start

### In Google Colab

```python
# Run the visualization demo
!python visualize_results.py

# Images will be automatically displayed in the notebook
```

### In Local Environment

```bash
# Generate visualization images
python visualize_results.py

# Or show interactively
python visualize_results.py --show
```

---

## 🎨 Visualization Types

The visualization script generates 5 different visualization types:

### 1. **3D Input Sparse Tensor** (`viz_1_input_3d.png`)

- Shows the spatial distribution of input points in 3D space
- Color indicates feature values
- Helps understand data sparsity

### 2. **3D Output Sparse Tensor** (`viz_2_output_3d.png`)

- Shows the model's output in 3D space
- Compare with input to see transformations
- Visualizes learned features

### 3. **Feature Distributions** (`viz_3_feature_dist.png`)

- Histograms for each feature channel
- Shows data distribution and range
- Helps identify outliers or patterns

### 4. **Input/Output Comparison** (`viz_4_comparison.png`)

- Side-by-side heatmaps
- Direct comparison of input vs output features
- Shows which points/features changed most

### 5. **Denoising Process** (`viz_5_denoising.png`)

- Shows denoising across multiple timesteps
- Visualizes the diffusion process in reverse
- 6 snapshots: t=999, 750, 500, 250, 100, 0

---

## 🌐 Google Colab Usage

### Complete Example

```python
# 1. Run the pipeline first (if not already done)
!python run_pipeline.py --resolution 32

# 2. Run visualization
!python visualize_results.py

# 3. Images will auto-display below!
```

### Custom Visualization in Colab

```python
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '/content/WaveMeshDf')

from models.unet_sparse import WaveMeshUNet
from models.spconv_compat import SparseConvTensor
from visualize_results import (
    visualize_sparse_tensor_3d,
    visualize_feature_distribution,
    visualize_prediction_comparison
)

# Load your model
model = WaveMeshUNet(
    in_channels=1,
    out_channels=1,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16],
    spatial_shape=(32, 32, 32)
).cuda()

# Create test data
batch_size = 1
num_points = 150
spatial_shape = (32, 32, 32)

indices = torch.randint(0, 32, (num_points, 3))
batch_col = torch.zeros((num_points, 1))
indices = torch.cat([batch_col, indices], dim=1).int().cuda()
features = torch.randn(num_points, 1).cuda()

sparse_input = SparseConvTensor(features, indices, spatial_shape, batch_size)

# Run model
model.eval()
with torch.no_grad():
    t = torch.tensor([500]).cuda()
    output = model(sparse_input, t)

# Visualize
fig = visualize_sparse_tensor_3d(output, "My Custom Output")
plt.show()
```

---

## 💻 Local Usage

### Basic Usage

```bash
# Generate all visualizations
python visualize_results.py

# Show interactively
python visualize_results.py --show
```

### Use in Your Scripts

```python
from visualize_results import (
    visualize_sparse_tensor_3d,
    visualize_feature_distribution,
    visualize_prediction_comparison,
    visualize_denoising_process
)

# Your model and data
model = ...
sparse_tensor = ...

# Generate visualizations
fig1 = visualize_sparse_tensor_3d(sparse_tensor, "My Data")
fig2 = visualize_feature_distribution(sparse_tensor)

# Save or show
fig1.savefig('my_visualization.png')
plt.show()
```

---

## 🎯 Custom Visualizations

### Visualize Specific Features

```python
# Visualize different feature channels
for i in range(output.features.shape[1]):
    fig = visualize_sparse_tensor_3d(
        output,
        f"Feature Channel {i}",
        feature_idx=i
    )
    plt.savefig(f'feature_{i}.png')
```

### Visualize Multiple Batches

```python
# If you have batch_size > 1
coords = output.indices.cpu().numpy()
features = output.features.cpu().numpy()

for batch_idx in range(batch_size):
    # Filter by batch
    mask = coords[:, 0] == batch_idx
    batch_coords = coords[mask]
    batch_features = features[mask]

    # Create sparse tensor for this batch
    batch_tensor = SparseConvTensor(
        torch.from_numpy(batch_features),
        torch.from_numpy(batch_coords).int(),
        output.spatial_shape,
        1
    )

    fig = visualize_sparse_tensor_3d(
        batch_tensor,
        f"Batch {batch_idx}"
    )
    plt.savefig(f'batch_{batch_idx}.png')
```

### Custom Timestep Visualization

```python
# Visualize specific timesteps
custom_timesteps = [999, 500, 100, 50, 10, 0]

fig = visualize_denoising_process(
    model,
    noisy_input,
    timesteps=custom_timesteps
)
plt.savefig('custom_denoising.png')
```

---

## 📸 Examples

### Example 1: Visualize Training Progress

```python
import torch
from visualize_results import visualize_feature_distribution

# During training
for epoch in range(num_epochs):
    # ... training code ...

    # Visualize every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            output = model(test_input, test_t)

        fig = visualize_feature_distribution(output)
        plt.savefig(f'epoch_{epoch}.png')
        plt.close()
```

### Example 2: Compare Different Resolutions

```python
resolutions = [16, 32, 64]

for res in resolutions:
    # Create data at different resolutions
    spatial_shape = (res, res, res)
    # ... create sparse tensor ...

    output = model(sparse_input, t)

    fig = visualize_sparse_tensor_3d(
        output,
        f"Resolution {res}³"
    )
    plt.savefig(f'resolution_{res}.png')
```

### Example 3: Visualize Real Mesh Data

```python
# After running Module A
from data.wavelet_utils import mesh_to_sdf_simple, sdf_to_sparse_wavelet
import trimesh

# Load mesh
mesh = trimesh.load('my_mesh.obj')

# Convert to SDF
sdf = mesh_to_sdf_simple(mesh, resolution=64)

# Convert to sparse wavelet
sparse_data = sdf_to_sparse_wavelet(sdf, threshold=0.01)

# Visualize
fig = visualize_sparse_tensor_3d(
    sparse_data,
    "Real Mesh - Sparse Wavelet"
)
plt.savefig('real_mesh.png')
```

---

## 🎨 Advanced: Interactive 3D in Colab

For interactive 3D plots in Colab:

```python
import plotly.graph_objects as go

# Extract data
coords = output.indices.cpu().numpy()
features = output.features.cpu().numpy()

x = coords[:, 1]
y = coords[:, 2]
z = coords[:, 3]
colors = features[:, 0]

# Create interactive plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=3,
        color=colors,
        colorscale='Viridis',
        showscale=True
    )
)])

fig.update_layout(
    title="Interactive 3D Sparse Tensor",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

fig.show()
```

---

## 📊 Output Statistics

The visualization script also prints statistics:

```
📊 Statistics:
   Input mean: 0.0234
   Input std: 0.9876
   Output mean: -0.0123
   Output std: 0.4567
```

These help you understand:

- Data normalization
- Feature scale
- Model behavior

---

## 🔧 Troubleshooting

### Issue: "No module named 'mpl_toolkits'"

```bash
# Install matplotlib with 3D support
pip install matplotlib
```

### Issue: Images not displaying in Colab

```python
# Manually display
from IPython.display import Image, display
display(Image('viz_1_input_3d.png'))
```

### Issue: Out of memory

```python
# Reduce number of points or resolution
num_points = 50  # Instead of 200
spatial_shape = (16, 16, 16)  # Instead of (32, 32, 32)
```

---

## 📚 API Reference

### `visualize_sparse_tensor_3d(sparse_tensor, title, feature_idx=0)`

Visualize sparse tensor in 3D space.

**Parameters:**

- `sparse_tensor`: SparseConvTensor object
- `title`: Plot title (str)
- `feature_idx`: Feature channel to visualize (int, default=0)

**Returns:** matplotlib Figure

---

### `visualize_feature_distribution(sparse_tensor, num_features=None)`

Show histogram of feature distributions.

**Parameters:**

- `sparse_tensor`: SparseConvTensor object
- `num_features`: Number of features to plot (int, default=min(16, total))

**Returns:** matplotlib Figure

---

### `visualize_prediction_comparison(input_features, output_features)`

Compare input and output features side-by-side.

**Parameters:**

- `input_features`: Input feature tensor (N x C)
- `output_features`: Output feature tensor (N x C)

**Returns:** matplotlib Figure

---

### `visualize_denoising_process(model, noisy_input, timesteps)`

Visualize denoising across timesteps.

**Parameters:**

- `model`: WaveMeshUNet model
- `noisy_input`: Initial noisy sparse tensor
- `timesteps`: List of timesteps to visualize (default=[999, 750, 500, 250, 100, 0])

**Returns:** matplotlib Figure

---

## ✅ Next Steps

After visualizing:

1. **Analyze Results**: Look for patterns in the visualizations
2. **Tune Parameters**: Adjust model settings based on what you see
3. **Train Model**: Use diffusion training with real data
4. **Generate Samples**: Use trained model to generate new meshes

---

## 📖 See Also

- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - How to run the pipeline
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation instructions
- [README.md](README.md) - Project overview

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue!
