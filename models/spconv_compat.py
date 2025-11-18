"""
Sparse Convolution Compatibility Layer
Provides fallback to dense operations when spconv is not available (e.g., Google Colab)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings

# Try to import spconv
try:
    import spconv.pytorch as spconv_original
    SPCONV_AVAILABLE = True
except (ImportError, Exception) as e:
    SPCONV_AVAILABLE = False
    spconv_original = None
    warnings.warn(
        f"spconv not available ({e}). Using dense fallback mode. "
        "Performance will be slower. For production, install spconv with GPU support.",
        UserWarning
    )


class SparseConvTensor:
    """
    Fallback SparseConvTensor that stores data densely
    Compatible with spconv.SparseConvTensor API
    """
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices  # [N, 4] where each row is [batch_idx, z, y, x]
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = None
        
    def dense(self):
        """Convert to dense tensor"""
        if self.grid is None:
            self._create_dense_grid()
        return self.grid
    
    def _create_dense_grid(self):
        """Create dense grid from sparse representation"""
        device = self.features.device
        C = self.features.shape[1]
        B = self.batch_size
        D, H, W = self.spatial_shape
        
        # Create empty grid
        self.grid = torch.zeros(B, C, D, H, W, device=device, dtype=self.features.dtype)
        
        # Fill in sparse values
        if len(self.indices) > 0:
            batch_idx = self.indices[:, 0].long()
            z_idx = self.indices[:, 1].long()
            y_idx = self.indices[:, 2].long()
            x_idx = self.indices[:, 3].long()
            
            self.grid[batch_idx, :, z_idx, y_idx, x_idx] = self.features
        
        return self.grid
    
    def replace_feature(self, new_features):
        """Replace features (API compatibility)"""
        self.features = new_features
        self.grid = None  # Invalidate cache
        return self
    
    def sample_at_indices(self, target_indices):
        """
        Resample this sparse tensor at the target indices.
        Used to align tensors with different sparse patterns.
        Returns a new SparseConvTensor with features sampled from this tensor's dense grid.
        """
        dense_grid = self.dense()
        B, C, D, H, W = dense_grid.shape
        
        features_list = []
        for i in range(len(target_indices)):
            b, z, y, x = target_indices[i].long()
            if 0 <= b < B and 0 <= z < D and 0 <= y < H and 0 <= x < W:
                feat = dense_grid[b, :, z, y, x]
            else:
                # Out of bounds - use zeros
                feat = torch.zeros(C, device=dense_grid.device, dtype=dense_grid.dtype)
            features_list.append(feat)
        
        if len(features_list) > 0:
            new_features = torch.stack(features_list, dim=0)
        else:
            new_features = torch.zeros((0, C), device=dense_grid.device, dtype=dense_grid.dtype)
        
        return SparseConvTensor(
            new_features, target_indices, self.spatial_shape, self.batch_size
        )


class DenseFallbackConv3d(nn.Module):
    """Dense 3D convolution fallback for SubMConv3d/SparseConv3d"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True, indice_key=None):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.indice_key = indice_key
        self.stride = stride
        
    def forward(self, input_tensor):
        if isinstance(input_tensor, SparseConvTensor):
            dense = input_tensor.dense()
            out_dense = self.conv(dense)
            
            B, C, D, H, W = out_dense.shape
            
            # For SubMConv (stride=1), preserve input indices exactly
            if self.stride == 1:
                # Extract features at the same indices as input
                indices = input_tensor.indices
                features_list = []
                
                for i in range(len(indices)):
                    b, z, y, x = indices[i].long()
                    # Handle potential out-of-bounds due to padding
                    if 0 <= z < D and 0 <= y < H and 0 <= x < W:
                        feat = out_dense[b, :, z, y, x]
                        features_list.append(feat)
                
                if len(features_list) > 0:
                    all_features = torch.stack(features_list, dim=0)
                else:
                    all_features = torch.zeros((0, C), device=out_dense.device)
                
                return SparseConvTensor(
                    all_features, indices, (D, H, W), B
                )
            else:
                # For SparseConv (stride>1), compute output indices from input indices
                # In sparse convolution with stride, output spatial dims are reduced
                # Output indices are: input_indices // stride (with floor division)
                indices = input_tensor.indices
                out_indices_list = []
                features_list = []
                
                for i in range(len(indices)):
                    b, z, y, x = indices[i].long()
                    # Compute strided output position
                    oz = z // self.stride
                    oy = y // self.stride
                    ox = x // self.stride
                    
                    # Check if within bounds
                    if 0 <= oz < D and 0 <= oy < H and 0 <= ox < W:
                        out_idx = torch.tensor([b, oz, oy, ox], device=indices.device, dtype=torch.long)
                        feat = out_dense[b, :, oz, oy, ox]
                        
                        out_indices_list.append(out_idx)
                        features_list.append(feat)
                
                if len(out_indices_list) > 0:
                    all_indices = torch.stack(out_indices_list, dim=0)
                    all_features = torch.stack(features_list, dim=0)
                    
                    # Remove duplicate indices (multiple input points can map to same output point)
                    # Keep the first occurrence (could also sum/max, but this is simpler)
                    unique_indices, inverse = torch.unique(all_indices, dim=0, return_inverse=True)
                    unique_features = torch.zeros(len(unique_indices), C, device=all_features.device, dtype=all_features.dtype)
                    
                    # Aggregate features for duplicate indices (sum them)
                    for i in range(len(inverse)):
                        unique_features[inverse[i]] += all_features[i]
                    
                    all_indices = unique_indices
                    all_features = unique_features
                else:
                    all_indices = torch.zeros((0, 4), device=out_dense.device, dtype=torch.long)
                    all_features = torch.zeros((0, C), device=out_dense.device)
                
                return SparseConvTensor(
                    all_features, all_indices, (D, H, W), B
                )
        else:
            return self.conv(input_tensor)


class DenseFallbackBatchNorm(nn.Module):
    """Dense BatchNorm fallback for spconv BatchNorm"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        
    def forward(self, input_tensor):
        if isinstance(input_tensor, SparseConvTensor):
            # Sparse tensor input - apply to features and return wrapped
            features = self.bn(input_tensor.features)
            return input_tensor.replace_feature(features)
        else:
            # Plain tensor input - just apply BatchNorm1d directly
            # This handles the case where code does: bn(sparse_tensor.features)
            return self.bn(input_tensor)


class DenseFallbackSequential(nn.Sequential):
    """Sequential module compatible with sparse tensors"""
    
    def forward(self, input_tensor):
        """Forward pass that handles both sparse and dense tensors"""
        x = input_tensor
        for module in self:
            # Check if this is an activation function or other element-wise operation
            if isinstance(x, SparseConvTensor):
                if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                    # Apply activation to features only
                    x = x.replace_feature(module(x.features))
                else:
                    # Regular module (conv, batchnorm, etc.)
                    x = module(x)
            else:
                x = module(x)
        return x


class DenseFallbackInverseConv3d(nn.Module):
    """Dense transposed convolution fallback for SparseInverseConv3d"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=2,
                 padding=0, bias=True, indice_key=None):
        super().__init__()
        # For inverse conv, output_padding helps match the expected output size
        output_padding = stride - 1 if stride > 1 else 0
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
            output_padding=output_padding
        )
        self.indice_key = indice_key
        self.stride = stride
        
    def forward(self, input_tensor):
        if isinstance(input_tensor, SparseConvTensor):
            dense = input_tensor.dense()
            out_dense = self.conv(dense)
            
            B, C, D, H, W = out_dense.shape
            
            # For upsampling, extract all non-zero locations from the output
            # This includes both upsampled input points and any new points created by the transposed conv
            mask = (out_dense.abs().sum(1) > 1e-10)
            
            indices_list = []
            features_list = []
            
            for b in range(B):
                nz = torch.nonzero(mask[b], as_tuple=False)  # [N, 3] (z, y, x)
                if len(nz) > 0:
                    batch_indices = torch.full((len(nz), 1), b, device=nz.device)
                    indices = torch.cat([batch_indices, nz], dim=1)  # [N, 4]
                    features = out_dense[b, :, nz[:, 0], nz[:, 1], nz[:, 2]].T  # [N, C]
                    
                    indices_list.append(indices)
                    features_list.append(features)
            
            if len(indices_list) > 0:
                all_indices = torch.cat(indices_list, dim=0)
                all_features = torch.cat(features_list, dim=0)
            else:
                all_indices = torch.zeros((0, 4), device=out_dense.device, dtype=torch.long)
                all_features = torch.zeros((0, C), device=out_dense.device)
            
            return SparseConvTensor(
                all_features, all_indices, (D, H, W), B
            )
        else:
            return self.conv(input_tensor)


# Create spconv-compatible API
if SPCONV_AVAILABLE:
    # Use real spconv
    SubMConv3d = spconv_original.SubMConv3d
    SparseConv3d = spconv_original.SparseConv3d
    SparseInverseConv3d = spconv_original.SparseInverseConv3d
    SparseConvTensor = spconv_original.SparseConvTensor
    SparseSequential = spconv_original.SparseSequential
    BatchNorm = lambda num_features, eps=1e-5, momentum=0.1: nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
else:
    # Use dense fallbacks
    SubMConv3d = DenseFallbackConv3d
    SparseConv3d = DenseFallbackConv3d
    SparseInverseConv3d = DenseFallbackInverseConv3d
    # SparseConvTensor already defined above
    SparseSequential = DenseFallbackSequential
    BatchNorm = DenseFallbackBatchNorm


def is_spconv_available():
    """Check if real spconv is available"""
    return SPCONV_AVAILABLE


def get_backend_info():
    """Get information about the backend being used"""
    if SPCONV_AVAILABLE:
        return {
            'backend': 'spconv',
            'version': getattr(spconv_original, '__version__', 'unknown'),
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'performance': 'optimal'
        }
    else:
        return {
            'backend': 'dense_fallback',
            'version': '1.0.0',
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'performance': 'suboptimal (10-100x slower)',
            'warning': 'Install spconv for better performance'
        }


# Convenience function to create sparse tensor
def sparse_tensor(features, indices, spatial_shape, batch_size):
    """Create a sparse tensor using the appropriate backend"""
    if SPCONV_AVAILABLE:
        return spconv_original.SparseConvTensor(
            features, indices, spatial_shape, batch_size
        )
    else:
        return SparseConvTensor(
            features, indices, spatial_shape, batch_size
        )
