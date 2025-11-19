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
        return_torch: bool = True,
        adaptive_threshold: bool = True,
        keep_approximation: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Convert dense SDF grid to sparse wavelet representation.
        
        Args:
            sdf_grid: Dense 3D SDF array of shape (D, H, W)
            threshold: Magnitude threshold for sparsification (coeffs below this are zeroed)
                      Set to 0.0 for lossless compression (no thresholding)
            return_torch: If True, return torch tensors instead of numpy arrays
            adaptive_threshold: If True, use different thresholds for approx vs detail coefficients
                              Approximation coeffs use threshold/10 (preserve more)
            keep_approximation: If True, always keep all approximation coefficients (recommended)
            
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
        
        # For multi-level DWT, we use wavedecn instead of manual iteration
        # This handles the multilevel structure correctly
        if self.level > 1:
            # Use wavedecn for proper multi-level decomposition
            coeffs_multilevel = pywt.wavedecn(sdf_grid, wavelet=self.wavelet, level=self.level, mode='periodization')
            # coeffs_multilevel is a list: [approx, {details_level_1}, {details_level_2}, ...]
            approx = coeffs_multilevel[0]  # Approximation at coarsest level
            details_levels = coeffs_multilevel[1:]  # List of detail coefficient dicts
            
            # Reconstruct the format for sparse conversion
            # We'll store all coefficients in a flat structure with level info
            all_coeffs_levels = []
            
            # Add coarsest approximation as a level
            all_coeffs_levels.append({'aaa': approx})
            
            # Add detail levels (from coarsest to finest)
            for detail_dict in details_levels:
                all_coeffs_levels.append(detail_dict)
        else:
            # Single level decomposition
            coeffs = pywt.dwtn(sdf_grid, wavelet=self.wavelet, mode='periodization')
            all_coeffs_levels = [coeffs]
        
        # Flatten all coefficient sub-bands into a single representation
        # Each sub-band becomes a channel in our sparse representation
        all_coeffs = []
        all_positions = []
        channel_info = []
        
        # Process all levels
        for level_idx, level_coeffs in enumerate(all_coeffs_levels):
            for key, coeff_array in sorted(level_coeffs.items()):
                # Determine threshold for this coefficient type
                is_approximation = key.count('a') == 3  # 'aaa' is approximation
                
                if threshold == 0.0:
                    # Lossless mode - keep all coefficients
                    current_threshold = 0.0
                elif keep_approximation and is_approximation:
                    # Keep all approximation coefficients (critical for reconstruction)
                    current_threshold = 0.0
                elif adaptive_threshold and is_approximation:
                    # Use lower threshold for approximation (preserve more)
                    current_threshold = threshold / 10.0
                else:
                    # Use standard threshold for detail coefficients
                    current_threshold = threshold
                
                # Apply threshold - sparsify
                mask = np.abs(coeff_array) > current_threshold
                
                if np.any(mask):
                    # Get coordinates where coefficients exceed threshold
                    indices = np.array(np.where(mask)).T  # Shape: (N_nonzero, 3)
                    values = coeff_array[mask]  # Shape: (N_nonzero,)
                    
                    all_positions.append(indices)
                    all_coeffs.append(values)
                    channel_info.append({
                        'key': key,
                        'level': level_idx,
                        'shape': coeff_array.shape,
                        'count': len(values),
                        'threshold': current_threshold,
                        'is_approximation': is_approximation
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
            'coeffs_levels': all_coeffs_levels,  # Keep all levels for reconstruction
            'channel_info': channel_info,
            'threshold': threshold,
            'adaptive_threshold': adaptive_threshold,
            'keep_approximation': keep_approximation,
            'original_sdf': sdf_grid  # Store original for residual correction (optional)
        }
        
        # Convert to PyTorch tensors if requested
        if return_torch:
            result['indices'] = torch.from_numpy(result['indices'])
            result['features'] = torch.from_numpy(result['features'])
        
        return result
    
    def sparse_to_dense_wavelet(
        self,
        sparse_data: Dict,
        denoise: bool = False,
        residual_correction: bool = False,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Reconstruct dense SDF grid from sparse wavelet representation.
        
        Args:
            sparse_data: Dictionary from dense_to_sparse_wavelet
            denoise: Whether to apply light denoising after reconstruction (reduces quality)
            residual_correction: Whether to apply residual correction using original SDF
                               Only works if original_sdf is stored in sparse_data
            verbose: Print debug information
            
        Returns:
            Reconstructed dense SDF grid
        """
        if verbose:
            print(f"Reconstruction config:")
            print(f"  - Wavelet: {sparse_data['wavelet']}")
            print(f"  - Levels: {sparse_data['level']}")
            print(f"  - Target shape: {sparse_data['shape']}")
            print(f"  - Sparse features: {len(sparse_data['features'])}")
        
        # Convert torch tensors to numpy if needed
        indices = sparse_data['indices']
        features = sparse_data['features']
        
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        # Reconstruct coefficient levels
        coeffs_levels = sparse_data['coeffs_levels']
        channel_info = sparse_data['channel_info']
        
        if verbose:
            print(f"  - Coefficient levels: {len(coeffs_levels)}")
            print(f"  - Channels: {len(channel_info)}")
        
        # Initialize empty coefficient arrays for all levels
        reconstructed_levels = []
        for level_idx, level_coeffs in enumerate(coeffs_levels):
            level_dict = {}
            for key, coeff_array in level_coeffs.items():
                level_dict[key] = np.zeros_like(coeff_array, dtype=np.float32)
                if verbose:
                    print(f"  - Level {level_idx}, key '{key}': shape {coeff_array.shape}")
            reconstructed_levels.append(level_dict)
        
        # Fill in the sparse values
        for ch_idx, ch_info in enumerate(channel_info):
            key = ch_info['key']
            level_idx = ch_info['level']
            
            # Get indices and values for this channel
            channel_mask = indices[:, 3] == ch_idx
            channel_indices = indices[channel_mask][:, :3]  # Remove channel dimension
            channel_values = features[channel_mask].flatten()
            
            # Place values back into dense array with bounds checking
            target_array = reconstructed_levels[level_idx][key]
            for (x, y, z), val in zip(channel_indices, channel_values):
                # Ensure indices are within bounds
                if 0 <= x < target_array.shape[0] and \
                   0 <= y < target_array.shape[1] and \
                   0 <= z < target_array.shape[2]:
                    target_array[x, y, z] = val
        
        # Perform inverse multi-level DWT
        # Reconstruct using wavedecn structure if multi-level
        num_levels = len(reconstructed_levels)
        
        if self.level > 1:
            # Use waverecn for proper multi-level reconstruction
            # Format: [approx, {details_level_n-1}, ..., {details_level_1}, {details_level_0}]
            coeffs_for_recon = []
            
            # First element is the approximation from coarsest level (level 0)
            approx_key = 'aaa'
            if approx_key in reconstructed_levels[0]:
                coeffs_for_recon.append(reconstructed_levels[0][approx_key])
            else:
                # Use first available key
                first_key = list(reconstructed_levels[0].keys())[0]
                coeffs_for_recon.append(reconstructed_levels[0][first_key])
            
            # Then add detail levels in order (from coarsest to finest)
            for level_idx in range(1, num_levels):
                # Remove the 'aaa' key if present (it's handled separately)
                detail_dict = {k: v for k, v in reconstructed_levels[level_idx].items() if k != 'aaa'}
                coeffs_for_recon.append(detail_dict)
            
            # Perform inverse multi-level wavelet transform
            try:
                reconstructed_sdf = pywt.waverecn(coeffs_for_recon, wavelet=self.wavelet, mode='periodization')
            except Exception as e:
                print(f"Warning: waverecn failed: {e}")
                # Fall back to single-level reconstruction
                reconstructed_sdf = pywt.idwtn(reconstructed_levels[0], wavelet=self.wavelet, mode='periodization')
        else:
            # Single level - use idwtn
            reconstructed_sdf = pywt.idwtn(reconstructed_levels[0], wavelet=self.wavelet, mode='periodization')
        
        
        # Ensure correct shape
        target_shape = sparse_data['shape']
        if reconstructed_sdf.shape != target_shape:
            # Crop or pad to match original shape
            reconstructed_sdf = self._match_shape(reconstructed_sdf, target_shape)
        
        # Optional residual correction (improves quality if original is available)
        if residual_correction and 'original_sdf' in sparse_data:
            original_sdf = sparse_data['original_sdf']
            # Compute residual error
            residual = original_sdf - reconstructed_sdf
            # Apply soft correction (reduce residual magnitude to avoid overcorrection)
            correction_strength = 0.5
            reconstructed_sdf = reconstructed_sdf + correction_strength * residual
        
        # Optional denoising (simple median filter) - WARNING: reduces quality
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
    padding: float = 0.1,
    method: str = 'auto'
) -> np.ndarray:
    """
    Convert mesh to dense SDF grid.
    
    Args:
        mesh_path: Path to mesh file (.obj, .ply, etc.)
        resolution: Grid resolution (creates resolution^3 grid)
        padding: Padding around mesh as fraction of bounding box
        method: SDF computation method - 'auto', 'scan', or 'simple'
                'auto': Try scan method, fall back to simple if headless
                'scan': Use mesh_to_sdf with scanning (requires display)
                'simple': Use simple distance-based method (works headless)
        
    Returns:
        Dense SDF grid of shape (resolution, resolution, resolution)
    """
    import trimesh
    
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
    
    # Compute SDF based on method
    if method == 'auto':
        try:
            # Try using mesh_to_sdf with scan method
            from mesh_to_sdf import mesh_to_sdf
            sdf_values = mesh_to_sdf(mesh, query_points, sign_method='normal')
        except Exception as e:
            # Fall back to simple method if scan fails (e.g., headless environment)
            print(f"  ⚠ mesh_to_sdf failed ({type(e).__name__}), using simple method...")
            sdf_values = _compute_sdf_simple(mesh, query_points)
    elif method == 'scan':
        from mesh_to_sdf import mesh_to_sdf
        sdf_values = mesh_to_sdf(mesh, query_points, sign_method='normal')
    elif method == 'simple':
        sdf_values = _compute_sdf_simple(mesh, query_points)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'scan', or 'simple'")
    
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    return sdf_grid


def _compute_sdf_simple(mesh, query_points: np.ndarray) -> np.ndarray:
    """
    Simple SDF computation using closest point on mesh surface.
    Works in headless environments. Less accurate than scan-based methods
    but sufficient for wavelet testing.
    
    Args:
        mesh: Trimesh object
        query_points: (N, 3) array of query points
        
    Returns:
        (N,) array of signed distances
    """
    import trimesh
    
    # Compute closest points on mesh surface
    closest_points, distances, triangle_id = mesh.nearest.on_surface(query_points)
    
    # Determine sign using ray casting (inside/outside test)
    # Count intersections along random rays - odd = inside, even = outside
    ray_directions = np.array([[1.0, 0.0, 0.0]])  # Use single direction for speed
    
    # Simple inside/outside test using winding number approximation
    # For a closed mesh, compute if point is inside using dot product with normals
    normals = mesh.face_normals[triangle_id]
    vectors_to_query = query_points - closest_points
    
    # If the vector from surface to query point is in the same direction as normal, it's outside
    dots = np.sum(vectors_to_query * normals, axis=1)
    signs = np.where(dots > 0, 1.0, -1.0)
    
    # For more robust inside/outside, use ray intersection count
    # But for simplicity and speed, the above approximation works for convex shapes
    try:
        # More robust method using contains check
        inside = mesh.contains(query_points)
        signs = np.where(inside, -1.0, 1.0)
    except:
        # Fall back to normal-based approximation
        pass
    
    sdf_values = signs * distances
    
    return sdf_values


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
    denoise: bool = False,
    residual_correction: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: sparse wavelet coefficients -> dense SDF -> mesh.
    
    Args:
        sparse_data: Dictionary from dense_to_sparse_wavelet
        level: Iso-surface level for marching cubes
        denoise: Whether to denoise during reconstruction (reduces quality)
        residual_correction: Apply residual correction for better quality
        
    Returns:
        vertices: (V, 3) array
        faces: (F, 3) array
    """
    # Initialize transformer with same parameters
    transformer = WaveletTransform3D(
        wavelet=sparse_data['wavelet'],
        level=sparse_data['level']
    )
    
    # Reconstruct SDF with improved quality
    sdf_grid = transformer.sparse_to_dense_wavelet(
        sparse_data, 
        denoise=denoise,
        residual_correction=residual_correction
    )
    
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


# ============================================================================
# Convenience Functions for Pipeline
# ============================================================================

def mesh_to_sdf_simple(mesh, resolution: int = 64) -> np.ndarray:
    """
    Simple convenience function: mesh object -> SDF grid.
    Works in headless environments (Colab).
    
    Args:
        mesh: Trimesh mesh object
        resolution: Grid resolution
        
    Returns:
        Dense SDF grid of shape (resolution, resolution, resolution)
    """
    import trimesh
    
    # Normalize mesh to unit cube
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max() * 1.1  # 10% padding
    
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(2.0 / scale)
    
    # Create query points
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Use simple SDF computation
    sdf_values = _compute_sdf_simple(mesh, query_points)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    return sdf_grid


def sdf_to_sparse_wavelet(
    sdf_grid: np.ndarray, 
    threshold: float = 0.01,
    wavelet: str = 'bior4.4',
    level: int = 3,
    adaptive_threshold: bool = True,
    keep_approximation: bool = True,
    lossless: bool = False
) -> Dict:
    """
    Convenience function: SDF grid -> sparse wavelet representation.
    
    Args:
        sdf_grid: Dense SDF array
        threshold: Sparsification threshold (ignored if lossless=True)
        wavelet: Wavelet type
        level: Decomposition level
        adaptive_threshold: Use adaptive thresholding (recommended)
        keep_approximation: Always keep approximation coefficients (recommended)
        lossless: If True, no thresholding (perfect reconstruction, but less sparse)
        
    Returns:
        Sparse wavelet dictionary
    """
    if lossless:
        threshold = 0.0
    
    transformer = WaveletTransform3D(wavelet=wavelet, level=level)
    return transformer.dense_to_sparse_wavelet(
        sdf_grid, 
        threshold=threshold, 
        return_torch=False,
        adaptive_threshold=adaptive_threshold,
        keep_approximation=keep_approximation
    )


def sparse_wavelet_to_sdf(
    sparse_data: Dict,
    resolution: int = None,
    denoise: bool = False,
    residual_correction: bool = False,
    verbose: bool = False
) -> np.ndarray:
    """
    Convenience function: sparse wavelet -> reconstructed SDF grid.
    
    Args:
        sparse_data: Dictionary from sdf_to_sparse_wavelet
        resolution: Target resolution (None = use original)
        denoise: Whether to apply denoising (reduces quality, use only if needed)
        residual_correction: Apply residual correction for better quality (if original available)
        verbose: Print reconstruction debug info
        
    Returns:
        Reconstructed dense SDF grid
    """
    transformer = WaveletTransform3D(
        wavelet=sparse_data['wavelet'],
        level=sparse_data['level']
    )
    return transformer.sparse_to_dense_wavelet(
        sparse_data, 
        denoise=denoise,
        residual_correction=residual_correction,
        verbose=verbose
    )


def normalize_mesh(mesh):
    """
    Normalize mesh to fit in unit cube centered at origin.
    
    Args:
        mesh: Trimesh mesh object
        
    Returns:
        Normalized mesh (new copy)
    """
    mesh = mesh.copy()
    
    # Center at origin
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    mesh.apply_translation(-center)
    
    # Scale to unit cube
    scale = (bounds[1] - bounds[0]).max()
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    
    return mesh
