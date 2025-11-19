# WaveMesh-Diff Project Structure

## Directory Organization

```
WaveMesh-Diff/
├── README.md                  # Main project documentation (copy from docs/)
│
├── docs/                      # All documentation files
│   ├── README.md             # Detailed project info
│   ├── DOCS.md               # Documentation index
│   ├── QUICKSTART.md         # Quick start guide
│   ├── ARCHITECTURE.md       # System architecture
│   ├── TRAINING.md           # Training instructions
│   ├── COLAB_SETUP.md        # Google Colab setup
│   ├── TROUBLESHOOTING.md    # Common issues & solutions
│   └── *.png                 # Result visualization images
│
├── examples/                  # Demo and example scripts
│   ├── demo_wavelet_improvement.py  # Wavelet quality demo
│   ├── quick_test.py                # Quick module tests
│   ├── run_pipeline.py              # Complete pipeline demo
│   └── visualize_results.py         # Result visualization
│
├── scripts/                   # Training and utility scripts
│   ├── train_colab.py        # Colab-optimized training
│   ├── train_colab.sh        # Training shell script
│   └── setup_headless.py     # Headless environment setup
│
├── tests/                     # Test files
│   ├── test_all_modules.py          # Comprehensive tests
│   ├── test_data_loading.py         # Data loading tests
│   ├── test_module_d.py             # MultiView encoder tests
│   ├── test_modules_bc.py           # U-Net & diffusion tests
│   ├── test_spconv_compat.py        # Spconv compatibility tests
│   ├── test_wavelet_pipeline.py     # Wavelet pipeline tests
│   └── test_wavelet_quality.py      # Wavelet quality tests
│
├── data/                      # Data processing modules
│   ├── __init__.py
│   ├── mesh_dataset.py       # Dataset loading
│   └── wavelet_utils.py      # Wavelet transforms
│
├── models/                    # Model definitions
│   ├── __init__.py
│   ├── diffusion.py          # Diffusion model (Module C)
│   ├── encoder.py            # MultiView encoder (Module D)
│   ├── spconv_compat.py      # Spconv compatibility layer
│   └── unet.py               # Sparse U-Net (Module B)
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── train_utils.py        # Training helpers
│
├── notebooks/                 # Jupyter notebooks
│   └── WaveMesh_Diff_Training.ipynb
│
├── train.py                   # Main training script
├── generate.py                # Generation script
└── requirements.txt           # Python dependencies
```

## Import Conventions

All subdirectory scripts (examples/, scripts/, tests/) use:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This allows importing from project root:

```python
from data.wavelet_utils import mesh_to_sdf_simple
from models.unet import SparseUNet
from utils.train_utils import get_optimizer
```

## Running Scripts

### From Project Root

```bash
# Training
python train.py --config config.yaml

# Examples
python examples/quick_test.py
python examples/demo_wavelet_improvement.py

# Tests
python tests/test_all_modules.py
python tests/test_wavelet_quality.py
```

### From Subdirectories

Scripts can also run from their own directories:

```bash
cd examples
python quick_test.py

cd ../tests
python test_all_modules.py
```

## Documentation Access

- **Main entry**: `README.md` in project root
- **Detailed docs**: `docs/` folder
- **Doc index**: `docs/DOCS.md`
- **Quick start**: `docs/QUICKSTART.md`
- **Training help**: `docs/TRAINING.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

## Recent Changes (2025-01-XX)

1. **Reorganized project structure**:

   - Moved all .md files → `docs/`
   - Moved demo scripts → `examples/`
   - Moved training utils → `scripts/`
   - Kept tests in `tests/`

2. **Fixed import paths**:

   - Added `sys.path` manipulation to all subdirectory scripts
   - All imports now reference from project root
   - Scripts work from any directory

3. **Updated documentation**:

   - README.md paths updated to point to `docs/`
   - DOCS.md references `../README.md`
   - All cross-references updated

4. **Cleaned up documentation** (12 → 7 files):
   - Removed: TRAINING_FIX.md, MEMORY_FIX.md, WAVELET_QUALITY.md, etc.
   - Consolidated into: TRAINING.md, TROUBLESHOOTING.md, COLAB_SETUP.md
