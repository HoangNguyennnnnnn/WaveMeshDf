"""
Test spconv compatibility layer
Verifies that dense fallback mode works correctly
"""

import torch
import numpy as np
from models.spconv_compat import (
    is_spconv_available, get_backend_info,
    SubMConv3d, SparseConv3d, SparseConvTensor,
    sparse_tensor
)


def test_backend_detection():
    """Test backend detection"""
    print("="*60)
    print("Test 1: Backend Detection")
    print("="*60)
    
    backend = get_backend_info()
    print(f"Backend: {backend['backend']}")
    print(f"Version: {backend['version']}")
    print(f"Device: {backend['device']}")
    print(f"Performance: {backend['performance']}")
    
    if is_spconv_available():
        print("✅ Using native spconv (optimal performance)")
    else:
        print("✅ Using dense fallback (functional, slower)")
    
    print()


def test_sparse_tensor_creation():
    """Test sparse tensor creation and conversion"""
    print("="*60)
    print("Test 2: Sparse Tensor Creation")
    print("="*60)
    
    # Create sparse data
    batch_size = 2
    num_points = 10
    num_channels = 4
    spatial_shape = (8, 8, 8)
    
    # Random sparse features
    features = torch.randn(num_points, num_channels)
    
    # Random sparse indices [batch_idx, z, y, x]
    indices = torch.randint(0, 8, (num_points, 3))
    batch_indices = torch.randint(0, batch_size, (num_points, 1))
    indices = torch.cat([batch_indices, indices], dim=1)
    
    # Create sparse tensor
    sp_tensor = sparse_tensor(features, indices, spatial_shape, batch_size)
    
    print(f"Created sparse tensor:")
    print(f"  - Features shape: {sp_tensor.features.shape}")
    print(f"  - Indices shape: {sp_tensor.indices.shape}")
    print(f"  - Spatial shape: {sp_tensor.spatial_shape}")
    print(f"  - Batch size: {sp_tensor.batch_size}")
    
    # Convert to dense
    dense = sp_tensor.dense()
    print(f"  - Dense shape: {dense.shape}")
    print(f"✅ Sparse tensor creation successful")
    print()


def test_sparse_convolution():
    """Test sparse convolution"""
    print("="*60)
    print("Test 3: Sparse Convolution")
    print("="*60)
    
    # Create sparse input
    batch_size = 1
    in_channels = 3
    out_channels = 8
    spatial_shape = (16, 16, 16)
    
    # Create some sparse points
    num_points = 50
    features = torch.randn(num_points, in_channels)
    indices = torch.cat([
        torch.zeros(num_points, 1, dtype=torch.long),  # batch 0
        torch.randint(0, 16, (num_points, 3))  # z, y, x
    ], dim=1)
    
    sp_input = sparse_tensor(features, indices, spatial_shape, batch_size)
    
    # Create convolution layer
    conv = SparseConv3d(in_channels, out_channels, kernel_size=3, padding=1)
    
    # Forward pass
    sp_output = conv(sp_input)
    
    print(f"Input:")
    print(f"  - Features: {sp_input.features.shape}")
    print(f"  - Channels: {in_channels}")
    
    print(f"Output:")
    print(f"  - Features: {sp_output.features.shape}")
    print(f"  - Channels: {out_channels}")
    
    assert sp_output.features.shape[1] == out_channels, "Output channels mismatch"
    print(f"✅ Sparse convolution successful")
    print()


def test_module_imports():
    """Test importing WaveMesh modules"""
    print("="*60)
    print("Test 4: Module Imports")
    print("="*60)
    
    try:
        from models import WaveMeshUNet, GaussianDiffusion
        print("✅ Successfully imported WaveMeshUNet")
        print("✅ Successfully imported GaussianDiffusion")
        
        # Create small model
        model = WaveMeshUNet(
            in_channels=1,
            out_channels=1,
            encoder_channels=[8, 16],
            use_attention=False
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Created WaveMeshUNet with {num_params:,} parameters")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        raise
    
    print()


def test_forward_pass():
    """Test complete forward pass"""
    print("="*60)
    print("Test 5: Complete Forward Pass")
    print("="*60)
    
    from models import WaveMeshUNet
    
    # Create tiny model
    encoder_channels = [4, 8]
    decoder_channels = [8, 4]  # Mirror the encoder
    
    model = WaveMeshUNet(
        in_channels=1,
        out_channels=1,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        use_attention=False
    )
    
    # Create sparse input
    batch_size = 1
    spatial_shape = (8, 8, 8)
    num_points = 20
    
    features = torch.randn(num_points, 1)
    indices = torch.cat([
        torch.zeros(num_points, 1, dtype=torch.long),
        torch.randint(0, 8, (num_points, 3))
    ], dim=1)
    
    sp_input = sparse_tensor(features, indices, spatial_shape, batch_size)
    
    # Forward pass
    print("Running forward pass...")
    timesteps = torch.zeros(batch_size, dtype=torch.long)  # Timestep 0
    
    with torch.no_grad():
        sp_output = model(sp_input, timesteps)
    
    print(f"Input features: {sp_input.features.shape}")
    print(f"Output features: {sp_output.features.shape}")
    print(f"✅ Forward pass successful")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WaveMesh-Diff Spconv Compatibility Test")
    print("="*60)
    print()
    
    test_backend_detection()
    test_sparse_tensor_creation()
    test_sparse_convolution()
    test_module_imports()
    test_forward_pass()
    
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print()
    
    backend = get_backend_info()
    if backend['backend'] == 'dense_fallback':
        print("ℹ️  Running in DENSE FALLBACK mode")
        print("   - Functional but slower (~10-50x)")
        print("   - Perfect for Google Colab!")
        print("   - For optimal performance, install spconv locally")
    else:
        print("⚡ Running with NATIVE SPCONV")
        print("   - Optimal performance!")
    print()
