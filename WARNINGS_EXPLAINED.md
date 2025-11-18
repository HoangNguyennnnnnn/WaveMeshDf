# Common Warnings Explained

This document explains common warnings you might see and whether they're important.

---

## ⚠️ "transformers not available. Using fallback encoder"

**What it means:** The `transformers` library (needed for DINOv2) is not installed.

**Is it a problem?**

- ❌ **NO** - for testing and development
- ⚠️ **YES** - for production (lower quality results)

**Solution:**

```bash
pip install transformers huggingface_hub
```

**Why it happens:**

- DINOv2 is a large pre-trained vision model that requires extra dependencies
- The code automatically falls back to a simpler CNN encoder
- Testing and development work fine with the fallback
- For best results, install transformers

---

## ⚠️ Debugger Warning: "frozen modules" / "PYDEVD_DISABLE_FILE_VALIDATION"

**Full warning:**

```
Debugger warning: It seems that frozen modules are being used, which may
make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
to python to disable frozen modules.
Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1
to disable this validation.
```

**What it means:** Python debugger warning about optimization features.

**Is it a problem?**

- ❌ **NO** - this is purely informational
- Code runs perfectly fine
- Only affects debugging/breakpoints

**Solution (if it bothers you):**

**Option 1: Environment Variable**

```bash
# Windows
set PYDEVD_DISABLE_FILE_VALIDATION=1
python your_script.py

# Linux/Mac
export PYDEVD_DISABLE_FILE_VALIDATION=1
python your_script.py
```

**Option 2: Python Flag**

```bash
python -Xfrozen_modules=off your_script.py
```

**Option 3: Use .pdbrc.py** (Already included in project)
The file `.pdbrc.py` in the project root automatically suppresses this warning.

---

## ⚠️ "kernel restarted"

**What it means:** Jupyter/notebook kernel was restarted.

**Is it a problem?**

- ❌ **NO** - normal notebook behavior
- Happens when installing packages or on errors
- Just re-run the cells after restart

---

## ⚠️ "spconv not available"

**What it means:** GPU-accelerated sparse convolution library not installed.

**Is it a problem?**

- ❌ **NO** - code works fine without it
- ⚠️ **Slightly slower** - uses dense operations instead

**Solution (optional):**

```bash
# For CUDA 11.8
pip install spconv-cu118

# For CUDA 11.7
pip install spconv-cu117
```

**Why it's optional:**

- spconv is GPU-specific and can be tricky to install
- Code automatically detects and uses it if available
- Falls back to standard PyTorch operations if not

---

## ⚠️ Import warnings from lint/type checker

**Examples:**

```
Import "torch" could not be resolved
Import "numpy" could not be resolved
```

**What it means:** VS Code/IDE can't find the package in the current environment.

**Is it a problem?**

- ❌ **NO** - if the code runs fine
- Just an IDE configuration issue
- Runtime imports work correctly

**Solution:**

1. Select correct Python interpreter in VS Code
2. Install packages in the active environment
3. Restart VS Code/IDE

---

## Summary: What to Ignore vs Fix

### ✅ Safe to Ignore

- Debugger frozen modules warning
- Kernel restart notifications
- IDE import warnings (if code runs)
- "spconv not available" (optional optimization)

### ⚠️ Should Fix for Production

- "transformers not available" → Install for better quality
- Missing core dependencies → Run `pip install -r requirements.txt`

### ❌ Must Fix

- "ModuleNotFoundError" for core packages (torch, numpy, pywt, trimesh)
- Actual runtime errors (not warnings)

---

## Quick Diagnostic

Run this to check your setup:

```python
import sys

print("="*60)
print("🔍 WaveMesh-Diff Installation Check")
print("="*60)

# Core dependencies
packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pywt': 'PyWavelets',
    'trimesh': 'Trimesh',
    'matplotlib': 'Matplotlib',
    'scipy': 'SciPy',
    'skimage': 'scikit-image'
}

print("\n✅ Core Dependencies:")
for module, name in packages.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - MISSING!")

# Optional dependencies
optional = {
    'transformers': 'Transformers (DINOv2)',
    'spconv': 'Sparse Convolutions (GPU)'
}

print("\n⚠️  Optional Dependencies:")
for module, name in optional.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ⚠️  {name} - not installed (OK)")

print("\n" + "="*60)
```

---

**Still seeing issues?** Check [INSTALLATION.md](INSTALLATION.md) or open an issue on GitHub.
