#!/usr/bin/env python3
"""
WaveMesh-Diff Result Visualization
Visualizes the outputs from the pipeline including:
- 3D sparse tensor data
- Feature distributions
- Model predictions
- Denoising process
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.unet_sparse import WaveMeshUNet
from models.spconv_compat import SparseConvTensor


def visualize_sparse_tensor_3d(sparse_tensor, title="Sparse Tensor Visualization", feature_idx=0):
    """
    Visualize a sparse tensor in 3D space
    
    Args:
        sparse_tensor: SparseConvTensor object
        title: Plot title
        feature_idx: Which feature channel to visualize (for color)
    """
    # Extract coordinates and features
    coords = sparse_tensor.indices.cpu().numpy()
    features = sparse_tensor.features.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get spatial coordinates (skip batch index)
    x = coords[:, 1]  # x coordinate
    y = coords[:, 2]  # y coordinate
    z = coords[:, 3]  # z coordinate
    
    # Color by feature value
    if features.shape[1] > feature_idx:
        colors = features[:, feature_idx]
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label=f'Feature {feature_idx}')
    else:
        ax.scatter(x, y, z, c='blue', s=20, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\n{len(coords)} points, {features.shape[1]} features')
    
    return fig


def visualize_feature_distribution(sparse_tensor, num_features=None):
    """
    Visualize the distribution of features in the sparse tensor
    
    Args:
        sparse_tensor: SparseConvTensor object
        num_features: Number of features to plot (default: all, max 16)
    """
    features = sparse_tensor.features.cpu().numpy()
    
    if num_features is None:
        num_features = min(features.shape[1], 16)
    
    # Create subplots
    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    for i in range(num_features):
        ax = axes[i]
        ax.hist(features[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Feature {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Distributions ({features.shape[0]} points, {features.shape[1]} channels)', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    return fig


def visualize_prediction_comparison(input_features, output_features):
    """
    Compare input and output features side by side
    
    Args:
        input_features: Input feature tensor
        output_features: Output feature tensor
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Input features
    if len(input_features.shape) == 2:
        im1 = axes[0].imshow(input_features.cpu().numpy().T, aspect='auto', cmap='viridis')
        axes[0].set_title(f'Input Features\n{input_features.shape[0]} points × {input_features.shape[1]} channels')
        axes[0].set_xlabel('Point Index')
        axes[0].set_ylabel('Feature Channel')
        plt.colorbar(im1, ax=axes[0])
    
    # Output features
    if len(output_features.shape) == 2:
        im2 = axes[1].imshow(output_features.cpu().numpy().T, aspect='auto', cmap='viridis')
        axes[1].set_title(f'Output Features\n{output_features.shape[0]} points × {output_features.shape[1]} channels')
        axes[1].set_xlabel('Point Index')
        axes[1].set_ylabel('Feature Channel')
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    return fig


def visualize_denoising_process(model, noisy_input, timesteps=[999, 750, 500, 250, 100, 0]):
    """
    Visualize the denoising process at different timesteps
    
    Args:
        model: WaveMeshUNet model
        noisy_input: Initial noisy sparse tensor
        timesteps: List of timesteps to visualize
    """
    model.eval()
    
    cols = 3
    rows = (len(timesteps) + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5*rows))
    
    with torch.no_grad():
        for idx, t in enumerate(timesteps):
            # Create time embedding
            t_tensor = torch.tensor([t], device=noisy_input.features.device)
            
            # Forward pass
            output = model(noisy_input, t_tensor)
            
            # Extract first feature for visualization
            coords = output.indices.cpu().numpy()
            features = output.features.cpu().numpy()
            
            # Create 3D subplot
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            
            x = coords[:, 1]
            y = coords[:, 2]
            z = coords[:, 3]
            colors = features[:, 0]
            
            scatter = ax.scatter(x, y, z, c=colors, cmap='coolwarm', s=15, alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Timestep t={t}')
            plt.colorbar(scatter, ax=ax, shrink=0.5)
    
    plt.suptitle('Denoising Process Over Time', fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig


def run_visualization_demo():
    """
    Run a complete visualization demo
    """
    print("="*70)
    print("🎨 WAVEMESH-DIFF VISUALIZATION DEMO")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Create model
    print("\n🧠 Creating WaveMeshUNet model...")
    model = WaveMeshUNet(
        in_channels=1,
        out_channels=1,
        encoder_channels=[16, 32, 64],
        decoder_channels=[64, 32, 16],
        use_attention=False  # Disable for faster testing
    ).to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test data
    print("\n📊 Creating test sparse data...")
    batch_size = 2
    num_points = 200
    spatial_shape = (32, 32, 32)
    
    # Random sparse coordinates
    indices = []
    for b in range(batch_size):
        batch_indices = torch.randint(0, min(spatial_shape), (num_points, 3))
        batch_col = torch.full((num_points, 1), b)
        batch_indices = torch.cat([batch_col, batch_indices], dim=1)
        indices.append(batch_indices)
    indices = torch.cat(indices, dim=0).int().to(device)
    
    # Random features
    features = torch.randn(batch_size * num_points, 1).to(device)
    
    # Create sparse tensor
    sparse_input = SparseConvTensor(features, indices, spatial_shape, batch_size)
    print(f"✅ Test data created: {len(indices)} points in {spatial_shape} space")
    
    # Run model
    print("\n🔄 Running forward pass...")
    model.eval()
    with torch.no_grad():
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        output = model(sparse_input, t)
    print(f"✅ Forward pass complete!")
    print(f"   Input: {sparse_input.features.shape}")
    print(f"   Output: {output.features.shape}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    # 1. 3D Visualization of Input
    print("   1️⃣  3D sparse tensor visualization...")
    fig1 = visualize_sparse_tensor_3d(sparse_input, "Input Sparse Tensor", feature_idx=0)
    plt.savefig('viz_1_input_3d.png', dpi=150, bbox_inches='tight')
    print("      Saved: viz_1_input_3d.png")
    
    # 2. 3D Visualization of Output
    print("   2️⃣  3D output visualization...")
    fig2 = visualize_sparse_tensor_3d(output, "Output Sparse Tensor", feature_idx=0)
    plt.savefig('viz_2_output_3d.png', dpi=150, bbox_inches='tight')
    print("      Saved: viz_2_output_3d.png")
    
    # 3. Feature Distributions
    print("   3️⃣  Feature distributions...")
    fig3 = visualize_feature_distribution(output, num_features=min(16, output.features.shape[1]))
    plt.savefig('viz_3_feature_dist.png', dpi=150, bbox_inches='tight')
    print("      Saved: viz_3_feature_dist.png")
    
    # 4. Input vs Output Comparison
    print("   4️⃣  Input/Output comparison...")
    fig4 = visualize_prediction_comparison(sparse_input.features, output.features)
    plt.savefig('viz_4_comparison.png', dpi=150, bbox_inches='tight')
    print("      Saved: viz_4_comparison.png")
    
    # 5. Denoising Process
    print("   5️⃣  Denoising process visualization...")
    fig5 = visualize_denoising_process(model, sparse_input, timesteps=[999, 750, 500, 250, 100, 0])
    plt.savefig('viz_5_denoising.png', dpi=150, bbox_inches='tight')
    print("      Saved: viz_5_denoising.png")
    
    # Show statistics
    print("\n📊 Statistics:")
    print(f"   Input mean: {sparse_input.features.mean().item():.4f}")
    print(f"   Input std: {sparse_input.features.std().item():.4f}")
    print(f"   Output mean: {output.features.mean().item():.4f}")
    print(f"   Output std: {output.features.std().item():.4f}")
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - viz_1_input_3d.png       (3D input visualization)")
    print("   - viz_2_output_3d.png      (3D output visualization)")
    print("   - viz_3_feature_dist.png   (Feature distributions)")
    print("   - viz_4_comparison.png     (Input/Output comparison)")
    print("   - viz_5_denoising.png      (Denoising process)")
    
    # Display in notebook if available
    try:
        import google.colab
        from IPython.display import Image, display
        print("\n🖼️  Displaying images...")
        for img in ['viz_1_input_3d.png', 'viz_2_output_3d.png', 'viz_3_feature_dist.png', 
                    'viz_4_comparison.png', 'viz_5_denoising.png']:
            display(Image(img))
    except:
        print("\n💡 To view images, check the generated PNG files above")
        print("   Or use: plt.show() in an interactive environment")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize WaveMesh-Diff results')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    run_visualization_demo()
    
    if args.show:
        plt.show()
