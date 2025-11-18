"""
Installation Verification Script for WaveMesh-Diff Module A
Run this script to verify all dependencies are correctly installed
"""

import sys
from typing import List, Tuple

def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Check if a module can be imported.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name (if different from module)
    
    Returns:
        Tuple of (success, version_or_error)
    """
    pkg_name = package_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, f"Not installed - run: pip install {pkg_name}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("=" * 80)
    print("WaveMesh-Diff Module A - Installation Verification")
    print("=" * 80)
    
    # Define required packages
    checks = [
        # Core
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        
        # Wavelet & 3D Processing
        ("pywt", "PyWavelets"),
        ("trimesh", "trimesh"),
        ("skimage", "scikit-image"),
        
        # Sparse Convolution (may fail if wrong CUDA version)
        ("spconv", "spconv-cu118"),
        
        # Additional
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("PIL", "pillow"),
    ]
    
    print("\nChecking required packages:\n")
    all_ok = True
    
    for module_name, package_name in checks:
        success, version = check_import(module_name, package_name)
        status = "✅" if success else "❌"
        print(f"{status} {package_name:20s} {version}")
        if not success:
            all_ok = False
    
    # Check mesh-to-sdf separately (different import)
    print("\nChecking optional packages:\n")
    try:
        import mesh_to_sdf
        print(f"✅ mesh-to-sdf          {getattr(mesh_to_sdf, '__version__', 'unknown')}")
    except ImportError:
        print(f"⚠️  mesh-to-sdf          Not installed - pip install mesh-to-sdf")
        print("   Note: Required for mesh_to_sdf_grid() function")
        all_ok = False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ transformers        {transformers.__version__}")
    except ImportError:
        print(f"⚠️  transformers        Not installed - pip install transformers")
        print("   Note: Will be required for Module D (Multi-view Encoder)")
    
    # Test local modules
    print("\n" + "=" * 80)
    print("Checking WaveMesh-Diff modules:")
    print("=" * 80 + "\n")
    
    try:
        from data.wavelet_utils import WaveletTransform3D
        print("✅ data.wavelet_utils   Successfully imported")
        print(f"   - WaveletTransform3D class available")
    except ImportError as e:
        print(f"❌ data.wavelet_utils   Import failed: {e}")
        all_ok = False
    
    try:
        from data import (
            mesh_to_sdf_grid,
            sdf_to_mesh,
            sparse_to_mesh,
            save_mesh,
            compute_sparsity,
            mesh_to_sdf_simple,
            sdf_to_sparse_wavelet,
            sparse_wavelet_to_sdf,
            normalize_mesh
        )
        print("✅ data module          All functions available")
        print(f"   - mesh_to_sdf_grid, mesh_to_sdf_simple, sdf_to_mesh")
        print(f"   - sdf_to_sparse_wavelet, sparse_wavelet_to_sdf")
        print(f"   - save_mesh, compute_sparsity, normalize_mesh")
    except ImportError as e:
        print(f"❌ data module          Import failed: {e}")
        all_ok = False
    
    # Quick functionality test
    print("\n" + "=" * 80)
    print("Running quick functionality test:")
    print("=" * 80 + "\n")
    
    try:
        import numpy as np
        from data.wavelet_utils import WaveletTransform3D
        
        # Create simple test SDF
        print("Creating test SDF grid (32³)...")
        test_sdf = np.random.randn(32, 32, 32).astype(np.float32)
        
        # Test wavelet transform
        print("Testing wavelet transform...")
        transformer = WaveletTransform3D(wavelet='bior4.4', level=2)
        sparse_data = transformer.dense_to_sparse_wavelet(
            test_sdf, 
            threshold=0.1, 
            return_torch=False
        )
        
        print(f"✅ Sparse representation created")
        print(f"   - Original size: {test_sdf.size:,} elements")
        print(f"   - Sparse size: {len(sparse_data['features']):,} coefficients")
        print(f"   - Sparsity: {(1 - len(sparse_data['features']) / test_sdf.size) * 100:.1f}%")
        
        # Test reconstruction
        print("Testing reconstruction...")
        reconstructed = transformer.sparse_to_dense_wavelet(sparse_data, denoise=False)
        
        mse = np.mean((test_sdf - reconstructed) ** 2)
        print(f"✅ Reconstruction successful")
        print(f"   - MSE: {mse:.6f}")
        
        if mse < 0.1:
            print(f"✅ Quality test passed (MSE < 0.1)")
        else:
            print(f"⚠️  High reconstruction error")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80 + "\n")
    
    if all_ok:
        print("✅ All checks passed! Module A is ready to use.\n")
        print("Next steps:")
        print("  1. Run: python tests/test_wavelet_pipeline.py --create-test-mesh")
        print("  2. Verify output meshes in ./output/ directory")
        print("  3. Check QUICKSTART.md for more testing options")
    else:
        print("❌ Some checks failed. Please install missing packages.\n")
        print("Quick fix:")
        print("  pip install -r requirements.txt")
        print("\nFor spconv issues, check your CUDA version:")
        print("  CUDA 11.8: pip install spconv-cu118")
        print("  CUDA 12.1: pip install spconv-cu121")
    
    print("=" * 80)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
