"""
Colab-optimized training wrapper for WaveMesh-Diff
Automatically detects memory constraints and adjusts settings
"""

import subprocess
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_system_info():
    """Get system memory and GPU info."""
    info = {
        'ram_gb': 0,
        'gpu_name': None,
        'gpu_memory_gb': 0
    }
    
    try:
        import psutil
        info['ram_gb'] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        print("⚠️  psutil not installed, cannot detect RAM")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split(',')
            info['gpu_name'] = gpu_info[0].strip()
            info['gpu_memory_gb'] = float(gpu_info[1].strip().split()[0]) / 1024
    except Exception:
        pass
    
    return info


def get_optimal_settings(ram_gb, gpu_memory_gb):
    """Determine optimal training settings based on available resources."""
    
    # Default: very conservative (works on Colab free tier)
    settings = {
        'resolution': 16,
        'batch_size': 4,
        'num_workers': 0,
        'unet_channels': [8, 16, 32],
        'diffusion_steps': 100,
    }
    
    # Adjust based on RAM
    if ram_gb >= 25:  # Colab Pro or local machine
        settings['resolution'] = 32
        settings['batch_size'] = 8
        settings['num_workers'] = 2
        settings['unet_channels'] = [16, 32, 64]
    elif ram_gb >= 16:  # Colab Pro
        settings['resolution'] = 24
        settings['batch_size'] = 6
        settings['num_workers'] = 1
        settings['unet_channels'] = [12, 24, 48]
    
    # Adjust based on GPU memory
    if gpu_memory_gb >= 16:  # A100, V100
        settings['batch_size'] = min(settings['batch_size'] * 2, 16)
        settings['diffusion_steps'] = 1000
    elif gpu_memory_gb >= 10:  # T4
        settings['diffusion_steps'] = 500
    
    return settings


def run_training(mode='debug', **custom_args):
    """
    Run training with optimal settings.
    
    Args:
        mode: 'debug' (fast test), 'quick' (1 epoch), or 'full' (complete training)
        **custom_args: Override any default arguments
    """
    
    # Detect system
    print("🔍 Detecting system resources...")
    info = get_system_info()
    
    print(f"\n📊 System Info:")
    print(f"  RAM: {info['ram_gb']:.1f} GB")
    if info['gpu_name']:
        print(f"  GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f} GB)")
    else:
        print(f"  GPU: None (CPU only)")
    
    # Get optimal settings
    settings = get_optimal_settings(info['ram_gb'], info['gpu_memory_gb'])
    
    print(f"\n⚙️  Optimal settings:")
    print(f"  Resolution: {settings['resolution']}³ voxels")
    print(f"  Batch size: {settings['batch_size']}")
    print(f"  Num workers: {settings['num_workers']}")
    print(f"  U-Net channels: {settings['unet_channels']}")
    print(f"  Diffusion steps: {settings['diffusion_steps']}")
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--data_root', custom_args.get('data_root', 'data/ModelNet40'),
        '--dataset', custom_args.get('dataset', 'modelnet40'),
        '--resolution', str(custom_args.get('resolution', settings['resolution'])),
        '--batch_size', str(custom_args.get('batch_size', settings['batch_size'])),
        '--num_workers', str(custom_args.get('num_workers', settings['num_workers'])),
        '--unet_channels', *map(str, custom_args.get('unet_channels', settings['unet_channels'])),
        '--diffusion_steps', str(custom_args.get('diffusion_steps', settings['diffusion_steps'])),
    ]
    
    # Mode-specific settings
    if mode == 'debug':
        cmd.extend([
            '--epochs', '5',
            '--max_samples', '20',
            '--output_dir', 'outputs/colab_debug',
            '--save_freq', '5',
            '--log_freq', '10',
        ])
        print(f"\n🚀 Starting DEBUG training (20 samples, 5 epochs, ~5 minutes)")
    
    elif mode == 'quick':
        cmd.extend([
            '--epochs', '1',
            '--max_samples', '100',
            '--output_dir', 'outputs/colab_quick',
            '--save_freq', '1',
            '--log_freq', '50',
        ])
        print(f"\n🚀 Starting QUICK training (100 samples, 1 epoch, ~15 minutes)")
    
    elif mode == 'full':
        cmd.extend([
            '--epochs', str(custom_args.get('epochs', 20)),
            '--output_dir', custom_args.get('output_dir', 'outputs/colab_full'),
            '--save_freq', str(custom_args.get('save_freq', 5)),
            '--log_freq', '100',
        ])
        print(f"\n🚀 Starting FULL training ({custom_args.get('epochs', 20)} epochs)")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'debug', 'quick', or 'full'")
    
    # Add custom overrides
    for key, value in custom_args.items():
        if key not in ['data_root', 'dataset', 'resolution', 'batch_size', 'num_workers', 
                       'unet_channels', 'diffusion_steps', 'epochs', 'output_dir', 'save_freq']:
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            elif isinstance(value, (list, tuple)):
                cmd.extend([f'--{key}', *map(str, value)])
            else:
                cmd.extend([f'--{key}', str(value)])
    
    print(f"\n📝 Command: {' '.join(cmd)}\n")
    print("=" * 60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"❌ Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Colab-optimized training for WaveMesh-Diff')
    parser.add_argument('--mode', type=str, default='debug', 
                        choices=['debug', 'quick', 'full'],
                        help='Training mode: debug (fast test), quick (1 epoch), full (complete)')
    parser.add_argument('--data_root', type=str, default='data/ModelNet40')
    parser.add_argument('--epochs', type=int, default=20, help='For full mode only')
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    custom_args = {}
    if args.data_root != 'data/ModelNet40':
        custom_args['data_root'] = args.data_root
    if args.mode == 'full':
        custom_args['epochs'] = args.epochs
        if args.output_dir:
            custom_args['output_dir'] = args.output_dir
    
    sys.exit(run_training(args.mode, **custom_args))
