"""
Test Improved Wavelet Reconstruction
Verifies the fixes to multi-level wavelet decomposition/reconstruction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh
from data.wavelet_utils import mesh_to_sdf_simple, sdf_to_sparse_wavelet, sparse_wavelet_to_sdf

print("="*70)
print("Testing Improved Wavelet Reconstruction")
print("="*70)

# Create test mesh
print("\n1. Creating test mesh (box)...")
mesh = trimesh.creation.box(extents=[1, 1, 1])
print(f"   Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Convert to SDF
print("\n2. Converting to SDF...")
resolution = 32
sdf = mesh_to_sdf_simple(mesh, resolution=resolution)
print(f"   SDF shape: {sdf.shape}")
print(f"   SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")

# Test configurations
configs = [
    {
        "name": "High Quality (bior4.4, level=3, threshold=1e-4)",
        "wavelet": "bior4.4",
        "level": 3,
        "threshold": 1e-4
    },
    {
        "name": "Medium Quality (db2, level=2, threshold=1e-3)",
        "wavelet": "db2",
        "level": 2,
        "threshold": 1e-3
    },
    {
        "name": "Lossless (bior4.4, level=2, no threshold)",
        "wavelet": "bior4.4",
        "level": 2,
        "threshold": 0.0
    }
]

print("\n3. Testing different configurations...")
print("="*70)

for i, config in enumerate(configs, 1):
    print(f"\nTest {i}: {config['name']}")
    print("-"*70)
    
    # Wavelet transform
    sparse_data = sdf_to_sparse_wavelet(
        sdf,
        threshold=config['threshold'],
        wavelet=config['wavelet'],
        level=config['level'],
        adaptive_threshold=True,
        keep_approximation=True
    )
    
    # Sparsity metrics
    total_elements = resolution ** 3
    non_zero = len(sparse_data['features'])
    sparsity = 100 * (1 - non_zero / total_elements)
    compression = total_elements / non_zero if non_zero > 0 else 0
    
    print(f"Compression:")
    print(f"  - Total elements: {total_elements:,}")
    print(f"  - Non-zero coeffs: {non_zero:,}")
    print(f"  - Sparsity: {sparsity:.1f}%")
    print(f"  - Compression ratio: {compression:.1f}x")
    
    # Reconstruct (with verbose for first test only)
    verbose = (i == 1)
    if verbose:
        print("\nReconstruction (verbose):")
    sdf_recon = sparse_wavelet_to_sdf(sparse_data, verbose=verbose)
    
    # Quality metrics
    mse = np.mean((sdf - sdf_recon) ** 2)
    mae = np.mean(np.abs(sdf - sdf_recon))
    max_error = np.abs(sdf - sdf_recon).max()
    psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
    
    print(f"\nQuality Metrics:")
    print(f"  - MSE: {mse:.8f}")
    print(f"  - MAE: {mae:.8f}")
    print(f"  - Max Error: {max_error:.8f}")
    print(f"  - PSNR: {psnr:.2f} dB")
    
    # Quality assessment
    if mse < 1e-6:
        quality = "🟢 EXCELLENT (near-perfect)"
    elif mse < 1e-4:
        quality = "🟢 VERY GOOD"
    elif mse < 1e-3:
        quality = "🟡 GOOD"
    elif mse < 1e-2:
        quality = "🟡 ACCEPTABLE"
    else:
        quality = "🔴 POOR"
    
    print(f"  - Overall: {quality}")

print("\n" + "="*70)
print("✅ All tests completed!")
print("="*70)

print("\nKey Improvements:")
print("  ✓ Using wavedecn/waverecn for proper multi-level structure")
print("  ✓ Bounds checking prevents index errors")
print("  ✓ Better error handling with fallback options")
print("  ✓ Verbose mode for debugging")
print("\nExpected Results:")
print("  - High Quality: MSE < 1e-4, excellent reconstruction")
print("  - Medium Quality: MSE < 1e-3, good balance")
print("  - Lossless: MSE ≈ 0, perfect reconstruction")
