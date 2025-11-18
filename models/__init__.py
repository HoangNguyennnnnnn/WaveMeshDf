"""
WaveMesh-Diff Models Module
Contains neural network architectures for diffusion-based mesh generation
"""

from .unet_sparse import (
    WaveMeshUNet,
    SparseResBlock,
    CrossAttentionBlock,
    SparseUNetEncoder,
    SparseUNetDecoder,
    create_sparse_tensor_from_wavelet
)

from .diffusion import GaussianDiffusion

from .spconv_compat import (
    SparseConvTensor,
    get_backend_info,
    sparse_tensor
)

__all__ = [
    'WaveMeshUNet',
    'SparseResBlock',
    'CrossAttentionBlock',
    'SparseUNetEncoder',
    'SparseUNetDecoder',
    'create_sparse_tensor_from_wavelet',
    'GaussianDiffusion',
    'SparseConvTensor',
    'get_backend_info',
    'sparse_tensor'
]
