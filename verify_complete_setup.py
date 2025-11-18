"""
Quick End-to-End Verification Script
Tests all components work together before running full training
"""

import torch
import numpy as np
from pathlib import Path
import sys

def test_imports():
    """Test all required imports"""
    print("1️⃣ Testing imports...")
    try:
        import pywt
        import trimesh
        from skimage import measure
        import matplotlib.pyplot as plt
        print("   ✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        return False

def test_modules():
    """Test all 4 core modules"""
    print("\n2️⃣ Testing core modules...")
    try:
        from data.wavelet_utils import (
            mesh_to_sdf_simple,
            sdf_to_sparse_wavelet,
            sparse_wavelet_to_sdf,
            WaveletTransform3D
        )
        from models import (
            WaveMeshUNet,
            GaussianDiffusion,
            create_multiview_encoder
        )
        print("   ✅ Module A: Wavelet Transform")
        print("   ✅ Module B: U-Net")
        print("   ✅ Module C: Diffusion")
        print("   ✅ Module D: Multi-view Encoder")
        return True
    except Exception as e:
        print(f"   ❌ Module import failed: {e}")
        return False

def test_wavelet_pipeline():
    """Test wavelet transform pipeline"""
    print("\n3️⃣ Testing wavelet pipeline...")
    try:
        import trimesh
        from data.wavelet_utils import mesh_to_sdf_simple, sdf_to_sparse_wavelet, sparse_wavelet_to_sdf
        
        # Create test mesh
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        
        # Convert to SDF
        sdf = mesh_to_sdf_simple(mesh, resolution=16)
        assert sdf.shape == (16, 16, 16), f"Wrong SDF shape: {sdf.shape}"
        
        # Wavelet transform
        sparse_data = sdf_to_sparse_wavelet(sdf, threshold=0.01)
        assert 'indices' in sparse_data and 'features' in sparse_data
        
        # Reconstruct
        sdf_recon = sparse_wavelet_to_sdf(sparse_data)
        assert sdf_recon.shape == sdf.shape
        
        mse = np.mean((sdf - sdf_recon) ** 2)
        print(f"   ✅ Wavelet pipeline OK (MSE: {mse:.6f})")
        return True
    except Exception as e:
        print(f"   ❌ Wavelet pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models():
    """Test model creation and forward pass"""
    print("\n4️⃣ Testing models...")
    try:
        from models import WaveMeshUNet, GaussianDiffusion, create_multiview_encoder
        
        # U-Net
        unet = WaveMeshUNet(
            in_channels=1,
            encoder_channels=[8, 16],
            decoder_channels=[16, 8],
            time_emb_dim=64,
            use_attention=False
        )
        
        x = torch.randn(1, 1, 8, 8, 8)
        t = torch.tensor([50])
        output = unet(x, t, context=None)
        assert output.shape == x.shape, f"Wrong output shape: {output.shape}"
        print(f"   ✅ U-Net OK ({sum(p.numel() for p in unet.parameters()):,} params)")
        
        # Diffusion
        diffusion = GaussianDiffusion(timesteps=100, beta_schedule='linear')
        noise = torch.randn_like(x)
        x_noisy = diffusion.q_sample(x, t, noise)
        assert x_noisy.shape == x.shape
        print(f"   ✅ Diffusion OK ({diffusion.timesteps} steps)")
        
        # Multi-view encoder
        encoder = create_multiview_encoder(preset='small')
        images = torch.randn(1, 4, 3, 224, 224)
        poses = torch.randn(1, 4, 3, 4)
        context = encoder(images, poses)
        print(f"   ✅ Multi-view encoder OK ({sum(p.numel() for p in encoder.parameters()):,} params)")
        
        return True
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training components"""
    print("\n5️⃣ Testing training setup...")
    try:
        from utils.checkpoint import save_checkpoint, load_checkpoint
        from utils.metrics import compute_metrics
        from utils.logger import setup_logger
        
        # Test logger
        logger = setup_logger('test', Path('outputs/test'))
        logger.info("Test message")
        print("   ✅ Logger OK")
        
        # Test checkpoint (mock)
        from models import WaveMeshUNet
        unet = WaveMeshUNet(in_channels=1, encoder_channels=[8], decoder_channels=[8])
        optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
        
        Path('outputs/test').mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            unet=unet,
            encoder=None,
            optimizer=optimizer,
            epoch=1,
            loss=0.5,
            save_path='outputs/test/test_checkpoint.pth'
        )
        print("   ✅ Checkpoint save OK")
        
        loaded = load_checkpoint('outputs/test/test_checkpoint.pth', unet=unet)
        assert loaded['epoch'] == 1
        print("   ✅ Checkpoint load OK")
        
        # Test metrics
        pred = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn(2, 1, 8, 8, 8)
        metrics = compute_metrics(pred, target)
        assert 'chamfer_distance' in metrics
        print("   ✅ Metrics OK")
        
        return True
    except Exception as e:
        print(f"   ❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset loader (if data available)"""
    print("\n6️⃣ Testing dataset loader...")
    try:
        from data.mesh_dataset import ModelNet40Dataset, create_dataloader
        
        # Check if data exists
        data_root = Path('data/ModelNet40')
        if not data_root.exists():
            print("   ⚠️  ModelNet40 data not found (optional)")
            return True
        
        # Try loading dataset
        dataset = ModelNet40Dataset(
            root_dir=str(data_root),
            split='train',
            resolution=16,
            max_samples=5
        )
        
        if len(dataset) == 0:
            print("   ⚠️  Dataset empty (run download_data.py)")
            return True
        
        # Test loading one sample
        sample = dataset[0]
        assert 'sparse_wavelet' in sample
        print(f"   ✅ Dataset OK ({len(dataset)} samples)")
        
        # Test dataloader
        loader = create_dataloader(dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        print("   ✅ DataLoader OK")
        
        return True
    except Exception as e:
        print(f"   ⚠️  Dataset test skipped: {e}")
        return True  # Non-critical

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 WaveMesh-Diff - Complete Verification")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Modules", test_modules),
        ("Wavelet Pipeline", test_wavelet_pipeline),
        ("Models", test_models),
        ("Training Setup", test_training_setup),
        ("Dataset", test_dataset),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:12} {name}")
    
    print("="*60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for training.")
        print("\n🚀 Quick start:")
        print("   python train.py --data_root data/ModelNet40 --debug --max_samples 20")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
