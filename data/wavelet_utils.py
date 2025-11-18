"""
Wavelet Utilities for WaveMesh-Diff
Handles conversion between dense SDF grids and sparse wavelet representations
"""

import numpy as np
import pywt
from typing import Tuple, Dict, Optional
from skimage import measure
import torch


class WaveletTransform3D:
    """
    3D Wavelet Transform handler for sparse 3D mesh generation.
    Uses biorthogonal wavelets for better reconstruction quality.
    """
    
    def __init__(self, wavelet: str = 'bior4.4', level: int = 3):
        """
        Args:
            wavelet: Wavelet family to use (default: bior4.4 for smooth reconstruction)
            level: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.level = level
        
    def dense_to_sparse_wavelet(
        self, 
        sdf_grid: np.ndarray, 
        threshold: float = 0.01,
        return_torch: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Convert dense SDF grid to sparse wavelet representation.
        
        Args:
            sdf_grid: Dense 3D SDF array of shape (D, H, W)
            threshold: Magnitude threshold for sparsification (coeffs below this are zeroed)
            return_torch: If True, return torch tensors instead of numpy arrays
            
        Returns:
            Dictionary containing:
                - 'indices': Sparse coordinate indices (N, 3) where N is number of non-zero coeffs
                - 'features': Corresponding wavelet coefficient values (N, C) where C is channels
                - 'shape': Original grid shape
                - 'level': Decomposition level used
                - 'wavelet': Wavelet type used
        """
        assert sdf_grid.ndim == 3, f"Expected 3D input, got shape {sdf_grid.shape}"
        original_shape = sdf_grid.shape
        
        # Perform 3D Discrete Wavelet Transform
        # dwtn returns a dictionary with approximation coeffs ('aaa') and detail coeffs
        coeffs = pywt.dwtn(sdf_grid, wavelet=self.wavelet, mode='periodization')
        
        # Multi-level decomposition
        for i in range(1, self.level):
            # Further decompose the approximation coefficients
            approx = coeffs['a' * 3]  # 'aaa' for 3D
            coeffs_next = pywt.dwtn(approx, wavelet=self.wavelet, mode='periodization')
            
            # Remove the old approximation and add new decomposition
            del coeffs['a' * 3]
            coeffs.update({f"L{i}_{k}": v for k, v in coeffs_next.items()})
        
        # Flatten all coefficient sub-bands into a single representation
        # Each sub-band becomes a channel in our sparse representation
        all_coeffs = []
        all_positions = []
        channel_info = []
        
        for key, coeff_array in sorted(coeffs.items()):
            # Apply threshold - sparsify
            mask = np.abs(coeff_array) > threshold
            
            if np.any(mask):
                # Get coordinates where coefficients exceed threshold
                indices = np.array(np.where(mask)).T  # Shape: (N_nonzero, 3)
                values = coeff_array[mask]  # Shape: (N_nonzero,)
                
                all_positions.append(indices)
                all_coeffs.append(values)
                channel_info.append({
                    'key': key,
                    'shape': coeff_array.shape,
                    'count': len(values)
                })
        
        # Concatenate all sparse coefficients
        if len(all_positions) == 0:
            # Handle edge case: completely empty sparse tensor
            sparse_indices = np.zeros((0, 4), dtype=np.int32)  # (batch, x, y, z)
            sparse_features = np.zeros((0, 1), dtype=np.float32)
        else:
            # Stack positions and values
            # We need to handle multi-scale coefficients properly
            # For simplicity, we'll concatenate them as separate channels
            sparse_indices = []
            sparse_features = []
            
            channel_offset = 0
            for idx, (positions, values) in enumerate(zip(all_positions, all_coeffs)):
                # Add channel dimension to indices
                # Format: (x, y, z, channel)
                channel_ids = np.full((len(positions), 1), channel_offset, dtype=np.int32)
                indices_with_channel = np.concatenate([positions, channel_ids], axis=1)
                
                sparse_indices.append(indices_with_channel)
                sparse_features.append(values.reshape(-1, 1))
                channel_offset += 1
            
            sparse_indices = np.concatenate(sparse_indices, axis=0).astype(np.int32)
            sparse_features = np.concatenate(sparse_features, axis=0).astype(np.float32)
        
        result = {
            'indices': sparse_indices,
            'features': sparse_features,
            'shape': original_shape,
            'level': self.level,
            'wavelet': self.wavelet,
            'coeffs_structure': coeffs,  # Keep original structure for reconstruction
            'channel_info': channel_info,
            'threshold': threshold
        }
        
        # Convert to PyTorch tensors if requested
        if return_torch:
            result['indices'] = torch.from_numpy(result['indices'])
            result['features'] = torch.from_numpy(result['features'])
        
        return result
    
    def sparse_to_dense_wavelet(
        self,
        sparse_data: Dict,
        denoise: bool = True
    ) -> np.ndarray:
        """
        Reconstruct dense SDF grid from sparse wavelet representation.
        
        Args:
            sparse_data: Dictionary from dense_to_sparse_wavelet
            denoise: Whether to apply light denoising after reconstruction
            
        Returns:
            Reconstructed dense SDF grid
        """
        # Convert torch tensors to numpy if needed
        indices = sparse_data['indices']
        features = sparse_data['features']
        
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        # Reconstruct coefficient dictionary
        coeffs_structure = sparse_data['coeffs_structure']
        channel_info = sparse_data['channel_info']
        
        # Initialize empty coefficient arrays
        reconstructed_coeffs = {}
        for key, coeff_array in coeffs_structure.items():
            reconstructed_coeffs[key] = np.zeros_like(coeff_array, dtype=np.float32)
        
        # Fill in the sparse values
        current_idx = 0
        for ch_idx, ch_info in enumerate(channel_info):
            key = ch_info['key']
            count = ch_info['count']
            
            # Get indices and values for this channel
            channel_mask = indices[:, 3] == ch_idx
            channel_indices = indices[channel_mask][:, :3]  # Remove channel dimension
            channel_values = features[channel_mask].flatten()
            
            # Place values back into dense array
            for (x, y, z), val in zip(channel_indices, channel_values):
                reconstructed_coeffs[key][x, y, z] = val
        
        # Perform inverse multi-level DWT
        # Reconstruct from finest to coarsest level
        current_coeffs = reconstructed_coeffs
        
        for i in range(self.level - 1, 0, -1):
            # Gather coefficients for this level
            level_coeffs = {k.split('_')[1]: v for k, v in current_coeffs.items() 
                          if k.startswith(f'L{i}_')}
            
            if level_coeffs:
                # Inverse transform
                approx = pywt.idwtn(level_coeffs, wavelet=self.wavelet, mode='periodization')
                
                # Remove processed level and add approximation
                current_coeffs = {k: v for k, v in current_coeffs.items() 
                                if not k.startswith(f'L{i}_')}
                current_coeffs[f'L{i-1}_aaa'] = approx
        
        # Final inverse transform
        final_coeffs = {k.split('_')[1] if '_' in k else k: v 
                       for k, v in current_coeffs.items()}
        
        reconstructed_sdf = pywt.idwtn(final_coeffs, wavelet=self.wavelet, mode='periodization')
        
        # Ensure correct shape
        target_shape = sparse_data['shape']
        if reconstructed_sdf.shape != target_shape:
            # Crop or pad to match original shape
            reconstructed_sdf = self._match_shape(reconstructed_sdf, target_shape)
        
        # Optional denoising (simple median filter)
        if denoise:
            from scipy.ndimage import median_filter
            reconstructed_sdf = median_filter(reconstructed_sdf, size=3)
        
        return reconstructed_sdf
    
    def _match_shape(self, array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Crop or pad array to match target shape."""
        result = np.zeros(target_shape, dtype=array.dtype)
        
        slices = tuple(slice(0, min(array.shape[i], target_shape[i])) for i in range(3))
        result[slices] = array[slices]
        
        return result


def mesh_to_sdf_grid(
    mesh_path: str,
    resolution: int = 256,
    padding: float = 0.1
) -> np.ndarray:
    """
    Convert mesh to dense SDF grid.
    
    Args:
        mesh_path: Path to mesh file (.obj, .ply, etc.)
        resolution: Grid resolution (creates resolution^3 grid)
        padding: Padding around mesh as fraction of bounding box
        
    Returns:
        Dense SDF grid of shape (resolution, resolution, resolution)
    """
    import trimesh
    from mesh_to_sdf import mesh_to_sdf
    
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Normalize mesh to unit cube
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max() * (1 + padding)
    
    mesh.apply_translation(-center)
    mesh.apply_scale(2.0 / scale)
    
    # Create query points
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Compute SDF
    sdf_values = mesh_to_sdf(mesh, query_points, sign_method='normal')
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    return sdf_grid


def sdf_to_mesh(
    sdf_grid: np.ndarray,
    level: float = 0.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from SDF grid using Marching Cubes.
    
    Args:
        sdf_grid: Dense SDF array
        level: Iso-surface level (0.0 for surface)
        spacing: Voxel spacing in each dimension
        
    Returns:
        vertices: (V, 3) array of vertex positions
        faces: (F, 3) array of triangle faces
    """
    # Run marching cubes
    vertices, faces, normals, _ = measure.marching_cubes(
        sdf_grid, 
        level=level,
        spacing=spacing,
        allow_degenerate=False
    )
    
    # Center and normalize vertices to [-1, 1]
    vertices = vertices - vertices.mean(axis=0)
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / scale
    
    return vertices, faces


def sparse_to_mesh(
    sparse_data: Dict,
    level: float = 0.0,
    denoise: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: sparse wavelet coefficients -> dense SDF -> mesh.
    
    Args:
        sparse_data: Dictionary from dense_to_sparse_wavelet
        level: Iso-surface level for marching cubes
        denoise: Whether to denoise during reconstruction
        
    Returns:
        vertices: (V, 3) array
        faces: (F, 3) array
    """
    # Initialize transformer with same parameters
    transformer = WaveletTransform3D(
        wavelet=sparse_data['wavelet'],
        level=sparse_data['level']
    )
    
    # Reconstruct SDF
    sdf_grid = transformer.sparse_to_dense_wavelet(sparse_data, denoise=denoise)
    
    # Extract mesh
    vertices, faces = sdf_to_mesh(sdf_grid, level=level)
    
    return vertices, faces


def save_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: str):
    """Save mesh to file using trimesh."""
    import trimesh
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")


def compute_sparsity(sparse_data: Dict) -> Dict[str, float]:
    """
    Compute sparsity statistics.
    
    Returns:
        Dictionary with sparsity metrics
    """
    total_elements = np.prod(sparse_data['shape'])
    non_zero_elements = len(sparse_data['features'])
    
    sparsity_ratio = 1.0 - (non_zero_elements / total_elements)
    compression_ratio = total_elements / non_zero_elements if non_zero_elements > 0 else float('inf')
    
    return {
        'total_elements': int(total_elements),
        'non_zero_elements': int(non_zero_elements),
        'sparsity_ratio': float(sparsity_ratio),
        'compression_ratio': float(compression_ratio),
        'memory_dense_mb': float(total_elements * 4 / 1024 / 1024),  # float32
        'memory_sparse_mb': float(non_zero_elements * 4 / 1024 / 1024),  # approximate
    }
