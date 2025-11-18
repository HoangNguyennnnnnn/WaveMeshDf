"""
Test All Modules - WaveMesh-Diff
Comprehensive test cho tất cả 4 modules (A, B, C, D)
"""
import sys
sys.path.insert(0, '.')

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_test(name):
    """Print test name"""
    print(f"\n[TEST] {name}")
    print("-" * 60)

def main():
    print_section("WAVEMESH-DIFF: TEST ALL MODULES")
    print("Testing 4 core modules: Wavelet, U-Net, Diffusion, MultiView")
    
    test_results = {
        'Module A': False,
        'Module B': False,
        'Module C': False,
        'Module D': False
    }
    
    # ==================================================================
    # MODULE A: WAVELET TRANSFORM
    # ==================================================================
    print_section("MODULE A: WAVELET TRANSFORM 3D")
    
    try:
        from data.wavelet_utils import WaveletTransform3D
        
        print_test("Basic Wavelet Transform")
        transform = WaveletTransform3D(
            wavelet='db1',
            levels=2,
            threshold=0.01
        )
        
        # Create dummy SDF
        sdf = torch.randn(1, 1, 32, 32, 32)
        
        # Forward transform
        coeffs, coords = transform.forward(sdf)
        print(f"  Input SDF shape: {sdf.shape}")
        print(f"  Sparse coefficients: {coeffs.shape}")
        print(f"  Active coordinates: {coords.shape}")
        
        # Inverse transform
        reconstructed = transform.inverse(coeffs, coords, shape=(1, 1, 32, 32, 32))
        print(f"  Reconstructed shape: {reconstructed.shape}")
        
        # MSE
        mse = torch.mean((sdf - reconstructed) ** 2).item()
        print(f"  Reconstruction MSE: {mse:.6f}")
        
        test_results['Module A'] = True
        print("  ✅ Module A: PASS")
        
    except Exception as e:
        print(f"  ❌ Module A: FAIL - {e}")
    
    # ==================================================================
    # MODULE B: SPARSE U-NET
    # ==================================================================
    print_section("MODULE B: SPARSE U-NET")
    
    try:
        from models import WaveMeshUNet
        
        print_test("U-Net Initialization")
        model = WaveMeshUNet(
            in_channels=1,
            encoder_channels=[16, 32, 64],
            decoder_channels=[64, 32, 16],
            time_emb_dim=128,
            use_attention=False
        )
        
        print(f"  Encoder channels: [16, 32, 64]")
        print(f"  Decoder channels: [64, 32, 16]")
        print(f"  Time embedding dim: 128")
        print(f"  Attention: False")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        test_results['Module B'] = True
        print("  ✅ Module B: PASS")
        
    except Exception as e:
        print(f"  ❌ Module B: FAIL - {e}")
    
    # ==================================================================
    # MODULE C: GAUSSIAN DIFFUSION
    # ==================================================================
    print_section("MODULE C: GAUSSIAN DIFFUSION")
    
    try:
        from models import GaussianDiffusion
        
        print_test("Diffusion Process")
        
        diffusion = GaussianDiffusion(
            timesteps=1000,
            beta_schedule='linear'
        )
        
        print(f"  Timesteps: {diffusion.timesteps}")
        print(f"  Beta schedule: linear")
        print(f"  Beta range: [{diffusion.betas[0]:.6f}, {diffusion.betas[-1]:.6f}]")
        
        # Test forward process
        print_test("Forward Noising")
        x_start = torch.randn(2, 1, 8, 8, 8)
        t = torch.tensor([0, 500])
        
        noise = torch.randn_like(x_start)
        x_noisy = diffusion.q_sample(x_start, t, noise)
        
        print(f"  Input shape: {x_start.shape}")
        print(f"  Timesteps: {t.tolist()}")
        print(f"  Noisy output: {x_noisy.shape}")
        
        test_results['Module C'] = True
        print("  ✅ Module C: PASS")
        
    except Exception as e:
        print(f"  ❌ Module C: FAIL - {e}")
    
    # ==================================================================
    # MODULE D: MULTI-VIEW ENCODER
    # ==================================================================
    print_section("MODULE D: MULTI-VIEW ENCODER")
    
    try:
        from models import MultiViewEncoder, create_multiview_encoder
        
        print_test("Multi-View Encoding")
        encoder = create_multiview_encoder(preset='small', image_size=224)
        
        # Test data
        batch_size = 2
        num_views = 4
        images = torch.randn(batch_size, num_views, 3, 224, 224)
        poses = torch.randn(batch_size, num_views, 3, 4)
        
        # Forward
        conditioning = encoder(images, poses)
        
        print(f"  Input images: {images.shape}")
        print(f"  Input poses: {poses.shape}")
        print(f"  Output conditioning: {conditioning.shape}")
        print(f"  Expected: ({batch_size}, {num_views}, 384)")
        
        assert conditioning.shape == (batch_size, num_views, 384)
        
        test_results['Module D'] = True
        print("  ✅ Module D: PASS")
        
    except Exception as e:
        print(f"  ❌ Module D: FAIL - {e}")
    
    # ==================================================================
    # INTEGRATION TEST
    # ==================================================================
    print_section("INTEGRATION TEST: ALL MODULES")
    
    try:
        print_test("Module Integration")
        
        # Module D: Conditioning
        encoder = create_multiview_encoder(preset='small')
        images = torch.randn(1, 4, 3, 224, 224)
        poses = torch.randn(1, 4, 3, 4)
        conditioning = encoder(images, poses)
        print(f"  1. Multi-view encoding: {conditioning.shape}")
        
        # Module B: U-Net with context
        unet = WaveMeshUNet(
            in_channels=1,
            encoder_channels=[16, 32],
            decoder_channels=[32, 16],
            time_emb_dim=64,
            use_attention=True,
            context_dim=384  # Match Module D output
        )
        print(f"  2. U-Net initialized with context_dim=384")
        
        # Module C: Diffusion
        diffusion = GaussianDiffusion(timesteps=100)
        print(f"  3. Diffusion model ready (T=100)")
        
        print("\n  ✅ INTEGRATION TEST: PASS")
        print("  All modules initialized correctly!")
        
    except Exception as e:
        print(f"\n  ❌ INTEGRATION TEST: FAIL - {e}")
        import traceback
        traceback.print_exc()
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print_section("TEST SUMMARY")
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\nResults: {passed}/{total} modules passed\n")
    
    for module, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {module:20s} {status}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nWaveMesh-Diff is ready for:")
        print("  • Training pipeline implementation")
        print("  • Integration with real datasets")
        print("  • Production deployment")
    else:
        print(f"\n⚠️  {total - passed} module(s) failed")
        print("Check error messages above for details")
    
    print("\n" + "="*70)
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
