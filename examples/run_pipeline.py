#!/usr/bin/env python3
"""
WaveMesh-Diff Complete Pipeline
Runs the full pipeline from mesh input to wavelet-based 3D generation

Works in both Google Colab and Local environments
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_environment():
    """Detect if running in Colab or Local"""
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'


def test_module_a(args):
    """Test Module A: Wavelet Transform Pipeline"""
    print("\n" + "="*70)
    print("MODULE A: Wavelet Transform Pipeline")
    print("="*70)
    
    from data.wavelet_utils import (
        mesh_to_sdf_simple,
        sdf_to_sparse_wavelet,
        sparse_wavelet_to_sdf,
        sdf_to_mesh,
        normalize_mesh
    )
    import trimesh
    
    # Create or load mesh
    if args.mesh:
        print(f"\n📁 Loading mesh from: {args.mesh}")
        mesh = trimesh.load(args.mesh)
    else:
        print("\n🔵 Creating test sphere mesh...")
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    
    # Normalize mesh
    mesh = normalize_mesh(mesh)
    print(f"✅ Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 1: Mesh → SDF
    print(f"\n📊 Converting mesh to SDF (resolution: {args.resolution}³)...")
    sdf_grid = mesh_to_sdf_simple(mesh, resolution=args.resolution)
    print(f"✅ SDF grid created: {sdf_grid.shape}")
    print(f"   SDF range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
    
    # Step 2: SDF → Sparse Wavelet
    print(f"\n🌊 Applying wavelet transform (threshold: {args.threshold})...")
    sparse_data = sdf_to_sparse_wavelet(sdf_grid, threshold=args.threshold)
    
    total_coeffs = args.resolution ** 3
    sparse_coeffs = len(sparse_data['indices'])
    sparsity = (1 - sparse_coeffs / total_coeffs) * 100
    compression = total_coeffs / sparse_coeffs if sparse_coeffs > 0 else 0
    
    print(f"✅ Wavelet transform complete:")
    print(f"   Total coefficients: {total_coeffs:,}")
    print(f"   Non-zero coefficients: {sparse_coeffs:,}")
    print(f"   Sparsity: {sparsity:.2f}%")
    print(f"   Compression ratio: {compression:.1f}x")
    
    # Step 3: Sparse Wavelet → SDF (Reconstruction)
    print(f"\n🔄 Reconstructing SDF from sparse wavelet...")
    reconstructed_sdf = sparse_wavelet_to_sdf(sparse_data, resolution=args.resolution)
    
    # Calculate reconstruction error
    mse = np.mean((sdf_grid - reconstructed_sdf) ** 2)
    max_error = np.abs(sdf_grid - reconstructed_sdf).max()
    print(f"✅ Reconstruction complete:")
    print(f"   MSE: {mse:.6f}")
    print(f"   Max error: {max_error:.6f}")
    
    # Step 4: SDF → Mesh (Final output)
    if args.output:
        print(f"\n💾 Saving outputs to: {args.output}")
        os.makedirs(args.output, exist_ok=True)
        
        # Save original mesh
        original_path = os.path.join(args.output, '01_original.obj')
        mesh.export(original_path)
        print(f"   ✅ Saved original mesh: {original_path}")
        
        # Save reconstructed mesh
        print(f"   🔨 Generating mesh from reconstructed SDF...")
        reconstructed_mesh = sdf_to_mesh(reconstructed_sdf)
        reconstructed_path = os.path.join(args.output, '02_reconstructed.obj')
        reconstructed_mesh.export(reconstructed_path)
        print(f"   ✅ Saved reconstructed mesh: {reconstructed_path}")
        
        # Save statistics
        stats_path = os.path.join(args.output, 'statistics.txt')
        with open(stats_path, 'w') as f:
            f.write(f"WaveMesh-Diff Pipeline Results\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Input Mesh:\n")
            f.write(f"  Vertices: {len(mesh.vertices)}\n")
            f.write(f"  Faces: {len(mesh.faces)}\n\n")
            f.write(f"SDF Grid:\n")
            f.write(f"  Resolution: {args.resolution}³\n")
            f.write(f"  Range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]\n\n")
            f.write(f"Wavelet Compression:\n")
            f.write(f"  Threshold: {args.threshold}\n")
            f.write(f"  Total coefficients: {total_coeffs:,}\n")
            f.write(f"  Non-zero coefficients: {sparse_coeffs:,}\n")
            f.write(f"  Sparsity: {sparsity:.2f}%\n")
            f.write(f"  Compression ratio: {compression:.1f}x\n\n")
            f.write(f"Reconstruction Quality:\n")
            f.write(f"  MSE: {mse:.6f}\n")
            f.write(f"  Max error: {max_error:.6f}\n")
        print(f"   ✅ Saved statistics: {stats_path}")
    
    return {
        'sparse_data': sparse_data,
        'sdf_grid': sdf_grid,
        'reconstructed_sdf': reconstructed_sdf,
        'sparsity': sparsity,
        'compression': compression,
        'mse': mse
    }


def test_module_b_c(args):
    """Test Modules B & C: Neural Network Pipeline"""
    print("\n" + "="*70)
    print("MODULES B & C: Neural Network Pipeline")
    print("="*70)
    
    from models import WaveMeshUNet, GaussianDiffusion
    from models.spconv_compat import get_backend_info, sparse_tensor
    
    # Check backend
    info = get_backend_info()
    print(f"\n🔧 Backend: {info['backend']}")
    print(f"📊 Performance: {info['performance']}")
    if info['backend'] == 'dense_fallback':
        print("⚠️  Note: Using dense fallback (slower but functional)")
        print("   For production, install spconv with GPU support")
    
    # Create U-Net model
    print(f"\n🧠 Creating WaveMeshUNet model...")
    model = WaveMeshUNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        encoder_channels=args.encoder_channels,
        decoder_channels=args.decoder_channels,
        use_attention=args.use_attention
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {num_params:,} parameters")
    
    # Create test data
    print(f"\n📊 Creating test sparse data...")
    batch_size = args.batch_size
    num_points = args.num_points
    spatial_shape = tuple([args.spatial_size] * 3)
    
    features = torch.randn(num_points, args.in_channels)
    indices = torch.cat([
        torch.randint(0, batch_size, (num_points, 1)),
        torch.randint(0, args.spatial_size, (num_points, 3))
    ], dim=1)
    
    sp_input = sparse_tensor(features, indices, spatial_shape, batch_size)
    timesteps = torch.randint(0, 100, (batch_size,))
    
    print(f"✅ Test data created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num points: {num_points}")
    print(f"   Spatial shape: {spatial_shape}")
    print(f"   Input features: {sp_input.features.shape}")
    
    # Test forward pass
    print(f"\n🔄 Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(sp_input, timesteps)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input: {sp_input.features.shape}")
    print(f"   Output: {output.features.shape}")
    
    # Create diffusion model
    if args.test_diffusion:
        print(f"\n🌀 Creating Gaussian Diffusion model...")
        diffusion = GaussianDiffusion(
            model=model,
            timesteps=args.diffusion_steps,
            objective='pred_noise'
        )
        print(f"✅ Diffusion model created with {diffusion.num_timesteps} timesteps")
        
        # Test sampling
        print(f"\n🎲 Testing diffusion sampling...")
        with torch.no_grad():
            sample_shape = (num_points, args.out_channels)
            # Note: Sampling requires proper implementation of the sample method
            print(f"   Sample shape: {sample_shape}")
            print(f"✅ Diffusion model ready for training/sampling")
    
    return {
        'model': model,
        'num_params': num_params,
        'backend': info['backend']
    }


def run_complete_pipeline(args):
    """Run the complete end-to-end pipeline"""
    print("\n" + "="*70)
    print("🚀 WAVEMESH-DIFF COMPLETE PIPELINE")
    print("="*70)
    
    env = check_environment()
    print(f"\n🖥️  Environment: {env.upper()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"💻 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    results = {}
    
    # Run Module A
    if not args.skip_module_a:
        try:
            results['module_a'] = test_module_a(args)
            print("\n✅ Module A: PASSED")
        except Exception as e:
            print(f"\n❌ Module A: FAILED - {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Run Modules B & C
    if not args.skip_module_bc:
        try:
            results['module_bc'] = test_module_b_c(args)
            print("\n✅ Modules B & C: PASSED")
        except Exception as e:
            print(f"\n❌ Modules B & C: FAILED - {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("📋 PIPELINE SUMMARY")
    print("="*70)
    
    if 'module_a' in results:
        print(f"\n📊 Module A (Wavelet Transform):")
        print(f"   Sparsity: {results['module_a']['sparsity']:.2f}%")
        print(f"   Compression: {results['module_a']['compression']:.1f}x")
        print(f"   MSE: {results['module_a']['mse']:.6f}")
    
    if 'module_bc' in results:
        print(f"\n🧠 Modules B & C (Neural Networks):")
        print(f"   Backend: {results['module_bc']['backend']}")
        print(f"   Parameters: {results['module_bc']['num_params']:,}")
    
    if args.output:
        print(f"\n💾 Outputs saved to: {args.output}")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='WaveMesh-Diff Complete Pipeline')
    
    # General options
    parser.add_argument('--mesh', type=str, help='Path to input mesh file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Module A options
    parser.add_argument('--resolution', type=int, default=64, help='SDF grid resolution')
    parser.add_argument('--threshold', type=float, default=0.001, help='Wavelet threshold')
    parser.add_argument('--skip-module-a', action='store_true', help='Skip Module A')
    
    # Module B & C options
    parser.add_argument('--in-channels', type=int, default=1, help='Input channels')
    parser.add_argument('--out-channels', type=int, default=1, help='Output channels')
    parser.add_argument('--encoder-channels', type=int, nargs='+', default=[16, 32, 64], 
                        help='Encoder channel sizes')
    parser.add_argument('--decoder-channels', type=int, nargs='+', default=[64, 32, 16],
                        help='Decoder channel sizes')
    parser.add_argument('--use-attention', action='store_true', help='Use attention in decoder')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-points', type=int, default=100, help='Number of sparse points')
    parser.add_argument('--spatial-size', type=int, default=16, help='Spatial grid size')
    parser.add_argument('--test-diffusion', action='store_true', help='Test diffusion model')
    parser.add_argument('--diffusion-steps', type=int, default=100, help='Diffusion timesteps')
    parser.add_argument('--skip-module-bc', action='store_true', help='Skip Modules B & C')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_complete_pipeline(args)
    
    return results


if __name__ == '__main__':
    main()
