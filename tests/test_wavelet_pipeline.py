"""
Test script for WaveMesh-Diff Wavelet Pipeline
Verifies mesh -> sparse wavelet -> mesh reconstruction quality

Note: For headless environments (Colab, servers without display):
    The script will automatically use a simple SDF computation method
    if the scan-based method fails.
"""

import numpy as np
import trimesh
import sys
import os
from pathlib import Path

# Configure for headless rendering if needed
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # or 'osmesa'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.wavelet_utils import (
    WaveletTransform3D,
    mesh_to_sdf_grid,
    sdf_to_mesh,
    sparse_to_mesh,
    save_mesh,
    compute_sparsity
)


def test_wavelet_roundtrip(
    mesh_path: str,
    resolution: int = 128,
    wavelet_level: int = 3,
    threshold: float = 0.01,
    output_dir: str = "./output"
):
    """
    Test complete pipeline: Mesh -> SDF -> Wavelet -> Sparse -> Dense -> Mesh
    
    Args:
        mesh_path: Path to input mesh file
        resolution: SDF grid resolution (lower for testing, increase for quality)
        wavelet_level: Number of wavelet decomposition levels
        threshold: Sparsification threshold
        output_dir: Directory to save outputs
    """
    print("=" * 80)
    print("WaveMesh-Diff Wavelet Pipeline Test")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load original mesh
    print("\n[Step 1] Loading original mesh...")
    original_mesh = trimesh.load(mesh_path, force='mesh')
    print(f"  ✓ Loaded mesh: {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")
    
    # Save normalized original
    original_output = os.path.join(output_dir, "01_original.obj")
    original_mesh.export(original_output)
    print(f"  ✓ Saved to: {original_output}")
    
    # Step 2: Convert mesh to SDF grid
    print(f"\n[Step 2] Converting mesh to SDF grid (resolution={resolution}^3)...")
    sdf_grid = mesh_to_sdf_grid(mesh_path, resolution=resolution)
    print(f"  ✓ SDF grid shape: {sdf_grid.shape}")
    print(f"  ✓ SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
    print(f"  ✓ Memory: {sdf_grid.nbytes / 1024 / 1024:.2f} MB")
    
    # Step 3: Convert dense SDF to sparse wavelet
    print(f"\n[Step 3] Converting to sparse wavelet (level={wavelet_level}, threshold={threshold})...")
    transformer = WaveletTransform3D(wavelet='bior4.4', level=wavelet_level)
    sparse_data = transformer.dense_to_sparse_wavelet(
        sdf_grid, 
        threshold=threshold,
        return_torch=False
    )
    
    # Print sparsity statistics
    stats = compute_sparsity(sparse_data)
    print(f"  ✓ Sparse representation created")
    print(f"    - Total elements: {stats['total_elements']:,}")
    print(f"    - Non-zero elements: {stats['non_zero_elements']:,}")
    print(f"    - Sparsity ratio: {stats['sparsity_ratio']:.2%}")
    print(f"    - Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"    - Memory (dense): {stats['memory_dense_mb']:.2f} MB")
    print(f"    - Memory (sparse): {stats['memory_sparse_mb']:.2f} MB")
    print(f"    - Number of channels: {len(sparse_data['channel_info'])}")
    
    # Step 4: Reconstruct dense SDF from sparse wavelet
    print("\n[Step 4] Reconstructing dense SDF from sparse wavelet...")
    reconstructed_sdf = transformer.sparse_to_dense_wavelet(sparse_data, denoise=True)
    print(f"  ✓ Reconstructed SDF shape: {reconstructed_sdf.shape}")
    print(f"  ✓ Reconstructed SDF range: [{reconstructed_sdf.min():.4f}, {reconstructed_sdf.max():.4f}]")
    
    # Compute reconstruction error
    mse = np.mean((sdf_grid - reconstructed_sdf) ** 2)
    mae = np.mean(np.abs(sdf_grid - reconstructed_sdf))
    max_error = np.abs(sdf_grid - reconstructed_sdf).max()
    
    print(f"\n  Reconstruction Quality:")
    print(f"    - MSE: {mse:.6f}")
    print(f"    - MAE: {mae:.6f}")
    print(f"    - Max Error: {max_error:.6f}")
    
    # Step 5: Extract mesh from original SDF
    print("\n[Step 5] Extracting mesh from original SDF...")
    vertices_orig, faces_orig = sdf_to_mesh(sdf_grid, level=0.0)
    mesh_from_sdf = trimesh.Trimesh(vertices=vertices_orig, faces=faces_orig)
    sdf_output = os.path.join(output_dir, "02_from_sdf.obj")
    save_mesh(vertices_orig, faces_orig, sdf_output)
    print(f"  ✓ Mesh: {len(vertices_orig)} vertices, {len(faces_orig)} faces")
    
    # Step 6: Extract mesh from reconstructed SDF
    print("\n[Step 6] Extracting mesh from reconstructed SDF...")
    vertices_recon, faces_recon = sparse_to_mesh(sparse_data, level=0.0, denoise=True)
    reconstructed_output = os.path.join(output_dir, "03_reconstructed.obj")
    save_mesh(vertices_recon, faces_recon, reconstructed_output)
    print(f"  ✓ Mesh: {len(vertices_recon)} vertices, {len(faces_recon)} faces")
    
    # Step 7: Compare meshes
    print("\n[Step 7] Comparing meshes...")
    mesh_recon = trimesh.Trimesh(vertices=vertices_recon, faces=faces_recon)
    
    # Compute mesh metrics
    print(f"\n  Original Mesh (from SDF):")
    print(f"    - Vertices: {len(mesh_from_sdf.vertices)}")
    print(f"    - Faces: {len(mesh_from_sdf.faces)}")
    print(f"    - Volume: {mesh_from_sdf.volume:.6f}")
    print(f"    - Surface Area: {mesh_from_sdf.area:.6f}")
    print(f"    - Watertight: {mesh_from_sdf.is_watertight}")
    
    print(f"\n  Reconstructed Mesh:")
    print(f"    - Vertices: {len(mesh_recon.vertices)}")
    print(f"    - Faces: {len(mesh_recon.faces)}")
    print(f"    - Volume: {mesh_recon.volume:.6f}")
    print(f"    - Surface Area: {mesh_recon.area:.6f}")
    print(f"    - Watertight: {mesh_recon.is_watertight}")
    
    # Volume difference
    if mesh_from_sdf.volume > 0:
        volume_diff = abs(mesh_from_sdf.volume - mesh_recon.volume) / mesh_from_sdf.volume
        print(f"\n  Volume difference: {volume_diff:.2%}")
    
    # Hausdorff distance (approximate)
    try:
        # Sample points on both meshes
        points_orig = mesh_from_sdf.sample(10000)
        points_recon = mesh_recon.sample(10000)
        
        # Compute one-way distances
        from scipy.spatial import cKDTree
        tree_orig = cKDTree(points_orig)
        tree_recon = cKDTree(points_recon)
        
        dist_to_orig, _ = tree_recon.query(points_orig)
        dist_to_recon, _ = tree_orig.query(points_recon)
        
        hausdorff_approx = max(dist_to_orig.max(), dist_to_recon.max())
        mean_dist = (dist_to_orig.mean() + dist_to_recon.mean()) / 2
        
        print(f"\n  Geometric Distance:")
        print(f"    - Approximate Hausdorff: {hausdorff_approx:.6f}")
        print(f"    - Mean distance: {mean_dist:.6f}")
    except Exception as e:
        print(f"\n  Could not compute geometric distance: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"✓ Sparsity: {stats['sparsity_ratio']:.2%}")
    print(f"✓ SDF reconstruction MSE: {mse:.6f}")
    print(f"✓ Mesh reconstruction quality: Good" if mse < 0.001 else "⚠ Mesh reconstruction quality: Check parameters")
    print(f"\nOutput files saved to: {output_dir}")
    print("  - 01_original.obj (normalized input)")
    print("  - 02_from_sdf.obj (from dense SDF)")
    print("  - 03_reconstructed.obj (from sparse wavelet)")
    print("=" * 80)
    
    return {
        'sparse_data': sparse_data,
        'sdf_grid': sdf_grid,
        'reconstructed_sdf': reconstructed_sdf,
        'stats': stats,
        'mse': mse,
        'mae': mae
    }


def test_different_thresholds(mesh_path: str, resolution: int = 128):
    """Test multiple threshold values to find optimal sparsity/quality tradeoff."""
    print("\n" + "=" * 80)
    print("Testing Different Threshold Values")
    print("=" * 80)
    
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    # Generate SDF once
    print(f"\nGenerating SDF grid (resolution={resolution}^3)...")
    sdf_grid = mesh_to_sdf_grid(mesh_path, resolution=resolution)
    
    transformer = WaveletTransform3D(wavelet='bior4.4', level=3)
    
    print(f"\n{'Threshold':<12} {'Sparsity':<12} {'Compression':<15} {'MSE':<12} {'MAE':<12}")
    print("-" * 70)
    
    for threshold in thresholds:
        # Convert to sparse
        sparse_data = transformer.dense_to_sparse_wavelet(
            sdf_grid, 
            threshold=threshold,
            return_torch=False
        )
        
        # Reconstruct
        reconstructed_sdf = transformer.sparse_to_dense_wavelet(sparse_data, denoise=True)
        
        # Compute metrics
        stats = compute_sparsity(sparse_data)
        mse = np.mean((sdf_grid - reconstructed_sdf) ** 2)
        mae = np.mean(np.abs(sdf_grid - reconstructed_sdf))
        
        print(f"{threshold:<12.4f} {stats['sparsity_ratio']:<12.2%} "
              f"{stats['compression_ratio']:<15.2f} {mse:<12.6f} {mae:<12.6f}")
    
    print("-" * 70)
    print("\nRecommendation: Use threshold ~0.01 for good quality/compression balance")


def create_simple_test_mesh(output_path: str = "./test_mesh.obj"):
    """Create a simple test mesh (sphere) for quick testing."""
    print(f"\nCreating simple test sphere at {output_path}...")
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.export(output_path)
    print(f"✓ Test mesh created: {len(sphere.vertices)} vertices, {len(sphere.faces)} faces")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WaveMesh-Diff Wavelet Pipeline")
    parser.add_argument("--mesh", type=str, default=None, help="Path to input mesh file")
    parser.add_argument("--resolution", type=int, default=128, help="SDF grid resolution")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition levels")
    parser.add_argument("--threshold", type=float, default=0.01, help="Sparsification threshold")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--test-thresholds", action="store_true", help="Test multiple thresholds")
    parser.add_argument("--create-test-mesh", action="store_true", help="Create a simple test mesh")
    
    args = parser.parse_args()
    
    # Create test mesh if requested
    if args.create_test_mesh or args.mesh is None:
        test_mesh_path = create_simple_test_mesh()
        if args.mesh is None:
            args.mesh = test_mesh_path
    
    # Verify mesh exists
    if not os.path.exists(args.mesh):
        print(f"Error: Mesh file not found: {args.mesh}")
        print("Run with --create-test-mesh to create a simple test sphere")
        sys.exit(1)
    
    # Run main test
    results = test_wavelet_roundtrip(
        mesh_path=args.mesh,
        resolution=args.resolution,
        wavelet_level=args.level,
        threshold=args.threshold,
        output_dir=args.output
    )
    
    # Test different thresholds if requested
    if args.test_thresholds:
        test_different_thresholds(args.mesh, resolution=args.resolution)
    
    print("\n✓ All tests completed successfully!")
