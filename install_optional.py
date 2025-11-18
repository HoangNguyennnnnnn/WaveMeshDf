"""
Install Optional Dependencies for WaveMesh-Diff
Installs transformers and other optional packages for better performance
"""

import subprocess
import sys

def install_package(package, name=None):
    """Install a package using pip"""
    name = name or package
    print(f"📦 Installing {name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"   ✅ {name} installed successfully")
        return True
    except Exception as e:
        print(f"   ❌ Failed to install {name}: {e}")
        return False

def main():
    print("="*60)
    print("🚀 Installing Optional Dependencies")
    print("="*60)
    print()
    
    packages = [
        ("transformers", "Transformers (for DINOv2)"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("accelerate", "Accelerate (for faster inference)"),
    ]
    
    results = []
    for package, name in packages:
        success = install_package(package, name)
        results.append((name, success))
        print()
    
    # Summary
    print("="*60)
    print("📊 Installation Summary")
    print("="*60)
    
    for name, success in results:
        status = "✅ INSTALLED" if success else "❌ FAILED"
        print(f"{status:15} {name}")
    
    installed = sum(1 for _, success in results if success)
    total = len(results)
    
    print()
    if installed == total:
        print(f"🎉 All {total} packages installed successfully!")
        print()
        print("🔥 Enhanced features now available:")
        print("   • DINOv2 vision encoder (better quality)")
        print("   • Faster model loading from Hugging Face")
        print("   • Accelerated inference")
    else:
        print(f"⚠️  {total - installed}/{total} packages failed to install")
        print("   Basic functionality will still work with fallback encoders")
    
    print()
    print("="*60)
    print("✅ Setup complete! Run test_all_modules.py to verify")
    print("="*60)

if __name__ == '__main__':
    main()
