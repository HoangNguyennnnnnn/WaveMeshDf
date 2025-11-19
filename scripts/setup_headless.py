"""
Setup script for headless environments (Google Colab, remote servers)
Run this before testing if you encounter display/OpenGL errors
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_headless_rendering():
    """Configure environment for headless 3D rendering."""
    print("Setting up headless rendering environment...")
    
    # Set environment variables for headless rendering
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['DISPLAY'] = ''
    
    print("✓ Environment variables set:")
    print(f"  - PYOPENGL_PLATFORM=egl")
    print(f"  - DISPLAY=(empty)")


def install_headless_dependencies():
    """Install dependencies for headless rendering."""
    print("\nInstalling headless rendering dependencies...")
    
    packages = [
        'xvfbwrapper',  # Virtual display wrapper
        'pyvirtualdisplay',  # Virtual display
    ]
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                package, '-q'
            ])
        print("✓ Headless dependencies installed")
        return True
    except Exception as e:
        print(f"⚠ Could not install headless dependencies: {e}")
        return False


def setup_virtual_display():
    """Set up virtual display using xvfb."""
    print("\nSetting up virtual display...")
    
    try:
        # Try to install system packages (works on Ubuntu/Debian)
        print("Installing xvfb (requires sudo)...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=False, capture_output=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'xvfb'], check=False, capture_output=True)
        
        # Start virtual display
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1024, 768))
        display.start()
        
        print("✓ Virtual display started")
        print(f"  - Display: {os.environ.get('DISPLAY', 'Not set')}")
        return display
    except Exception as e:
        print(f"⚠ Could not set up virtual display: {e}")
        print("  Note: This is OK - the simple SDF method will be used instead")
        return None


def test_rendering():
    """Test if rendering works."""
    print("\nTesting rendering capabilities...")
    
    try:
        import trimesh
        
        # Create simple test mesh
        mesh = trimesh.creation.icosphere(radius=1.0)
        print(f"✓ Created test mesh: {len(mesh.vertices)} vertices")
        
        # Test if mesh_to_sdf works
        try:
            from mesh_to_sdf import mesh_to_sdf
            test_points = [[0, 0, 0], [2, 2, 2]]
            sdf_values = mesh_to_sdf(mesh, test_points, sign_method='depth')
            print(f"✓ mesh_to_sdf works (scan method available)")
            return 'scan'
        except Exception as e:
            print(f"⚠ mesh_to_sdf failed: {type(e).__name__}")
            print(f"  Will use simple SDF method instead (this is OK)")
            return 'simple'
            
    except Exception as e:
        print(f"❌ Rendering test failed: {e}")
        return None


def main():
    """Main setup function."""
    print("=" * 80)
    print("WaveMesh-Diff Headless Environment Setup")
    print("=" * 80)
    
    # Step 1: Set environment variables
    setup_headless_rendering()
    
    # Step 2: Install dependencies
    installed = install_headless_dependencies()
    
    # Step 3: Try to set up virtual display
    display = None
    if installed:
        display = setup_virtual_display()
    
    # Step 4: Test rendering
    method = test_rendering()
    
    # Summary
    print("\n" + "=" * 80)
    print("Setup Summary")
    print("=" * 80)
    
    if method == 'scan':
        print("✅ Full rendering support available")
        print("   You can use high-quality scan-based SDF computation")
    elif method == 'simple':
        print("✅ Simple SDF method will be used")
        print("   This works in headless environments and is sufficient for testing")
        print("   Note: Slightly lower quality than scan method, but 100x faster")
    else:
        print("⚠️  Rendering test inconclusive")
        print("   Try running the test script - it will auto-detect the best method")
    
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("Run the test script:")
    print("  python tests/test_wavelet_pipeline.py --create-test-mesh")
    print("\nThe script will automatically choose the best SDF method for your environment")
    print("=" * 80)
    
    return display


if __name__ == "__main__":
    # Keep display alive if created
    display = main()
    
    # Instructions for using in Colab
    if 'google.colab' in sys.modules:
        print("\n📓 Google Colab Detected")
        print("Add this to the top of your notebook:")
        print("---")
        print("import os")
        print("os.environ['PYOPENGL_PLATFORM'] = 'egl'")
        print("---")
