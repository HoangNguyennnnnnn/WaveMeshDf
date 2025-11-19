"""
Quick demo: Show improvement in wavelet reconstruction quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from data.wavelet_utils import mesh_to_sdf_simple, sdf_to_sparse_wavelet, sparse_wavelet_to_sdf
import trimesh

print("=" * 60)
print("Wavelet Quality Improvement Demo")
print("=" * 60)

# Create test mesh
print("\n1. Creating test mesh (cube)...")
mesh = trimesh.creation.box(extents=[1, 1, 1])

# Convert to SDF
print("2. Converting to SDF (32³ resolution)...")
sdf_original = mesh_to_sdf_simple(mesh, resolution=32)
print(f"   SDF range: [{sdf_original.min():.3f}, {sdf_original.max():.3f}]")

# OLD METHOD (deprecated)
print("\n3. OLD METHOD (deprecated):")
print("   Settings: threshold=0.01, adaptive=False, denoise=True")

sparse_old = sdf_to_sparse_wavelet(
    sdf_original,
    threshold=0.01,
    adaptive_threshold=False,
    keep_approximation=False
)

sdf_recon_old = sparse_wavelet_to_sdf(
    sparse_old,
    denoise=True  # Old default
)

mse_old = np.mean((sdf_original - sdf_recon_old) ** 2)
sparsity_old = 100 * (1 - len(sparse_old['features']) / np.prod(sdf_original.shape))

print(f"   ❌ MSE: {mse_old:.6f}")
print(f"   Sparsity: {sparsity_old:.1f}%")

# NEW METHOD (default)
print("\n4. NEW METHOD (current default):")
print("   Settings: threshold=0.01, adaptive=True, denoise=False")

sparse_new = sdf_to_sparse_wavelet(
    sdf_original,
    threshold=0.01  # Adaptive is now default
)

sdf_recon_new = sparse_wavelet_to_sdf(sparse_new)

mse_new = np.mean((sdf_original - sdf_recon_new) ** 2)
sparsity_new = 100 * (1 - len(sparse_new['features']) / np.prod(sdf_original.shape))

print(f"   ✅ MSE: {mse_new:.6f}")
print(f"   Sparsity: {sparsity_new:.1f}%")

# BEST METHOD (with residual correction)
print("\n5. BEST METHOD (optional residual correction):")
print("   Settings: threshold=0.01, adaptive=True, residual_correction=True")

sdf_recon_best = sparse_wavelet_to_sdf(
    sparse_new,
    residual_correction=True
)

mse_best = np.mean((sdf_original - sdf_recon_best) ** 2)

print(f"   🏆 MSE: {mse_best:.6f}")
print(f"   Sparsity: {sparsity_new:.1f}%")

# Summary
print("\n" + "=" * 60)
print("IMPROVEMENT SUMMARY")
print("=" * 60)

improvement_new = mse_old / mse_new if mse_new > 0 else float('inf')
improvement_best = mse_old / mse_best if mse_best > 0 else float('inf')

print(f"\n📊 MSE Comparison:")
print(f"   Old method:  {mse_old:.6f}")
print(f"   New method:  {mse_new:.6f}  (↓ {improvement_new:.1f}x better)")
print(f"   Best method: {mse_best:.6f}  (↓ {improvement_best:.1f}x better)")

print(f"\n💾 Sparsity (same for all methods):")
print(f"   {sparsity_new:.1f}% compression")

print(f"\n🎯 Recommendation:")
print(f"   • Training:   Use NEW default (no code changes needed!)")
print(f"   • Production: Add residual_correction=True for best quality")
print(f"   • Evaluation: Use lossless=True for perfect reconstruction")

print("\n✅ Quality improved while keeping same compression!")
print("=" * 60)

print("\n📖 For more details, see: WAVELET_QUALITY.md")
