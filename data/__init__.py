"""
WaveMesh-Diff Data Module
Contains wavelet transform utilities and dataset loaders
"""

from .wavelet_utils import (
    WaveletTransform3D,
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

__all__ = [
    'WaveletTransform3D',
    'mesh_to_sdf_grid',
    'sdf_to_mesh',
    'sparse_to_mesh',
    'save_mesh',
    'compute_sparsity',
    'mesh_to_sdf_simple',
    'sdf_to_sparse_wavelet',
    'sparse_wavelet_to_sdf',
    'normalize_mesh'
]
