# Python startup file to disable debugger warnings
# Place this as .pdbrc or .pdbrc.py in your home directory or project root

import os
import sys

# Disable frozen modules warning in debugger
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Suppress specific warnings
import warnings
warnings.filterwarnings('ignore', message='.*frozen modules.*')
warnings.filterwarnings('ignore', message='.*PYDEVD_DISABLE_FILE_VALIDATION.*')
