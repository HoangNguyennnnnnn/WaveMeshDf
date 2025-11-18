"""
Mesh Dataset for WaveMesh-Diff
Loads meshes from ModelNet40/ShapeNet and converts to SDF + sparse wavelet
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import trimesh
from typing import Optional, List, Tuple, Dict
import json
from tqdm import tqdm
import random

from .wavelet_utils import (
    mesh_to_sdf_simple,
    sdf_to_sparse_wavelet,
    normalize_mesh
)


class ModelNet40Dataset(Dataset):
    """
    ModelNet40 dataset for 3D mesh generation.
    
    Structure:
        data/ModelNet40/
            airplane/
                train/
                    airplane_0001.off
                    ...
                test/
                    ...
            chair/
                train/
                test/
            ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 32,
        wavelet_threshold: float = 0.01,
        wavelet_type: str = 'bior4.4',
        wavelet_level: int = 3,
        categories: Optional[List[str]] = None,
        transform=None,
        cache_sdf: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            root_dir: Path to ModelNet40 directory
            split: 'train' or 'test'
            resolution: SDF grid resolution (32, 64, 128)
            wavelet_threshold: Sparsification threshold
            wavelet_type: Wavelet family
            wavelet_level: Decomposition levels
            categories: List of categories to use (None = all)
            transform: Optional transform for data augmentation
            cache_sdf: Cache computed SDFs to disk
            max_samples: Limit number of samples (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.resolution = resolution
        self.wavelet_threshold = wavelet_threshold
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.transform = transform
        self.cache_sdf = cache_sdf
        
        # Find all mesh files
        self.samples = []
        all_categories = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        if categories is None:
            categories = all_categories
        
        for cat in categories:
            cat_path = self.root_dir / cat / split
            if not cat_path.exists():
                print(f"⚠️  Category {cat}/{split} not found, skipping")
                continue
            
            meshes = list(cat_path.glob("*.off"))
            for mesh_path in meshes:
                self.samples.append({
                    'path': mesh_path,
                    'category': cat,
                    'category_idx': all_categories.index(cat)
                })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        self.num_categories = len(all_categories)
        self.categories = all_categories
        
        print(f"📊 ModelNet40 {split}: {len(self.samples)} samples from {len(categories)} categories")
        
        # Cache directory
        if cache_sdf:
            self.cache_dir = self.root_dir / f"cache_sdf_{resolution}"
            self.cache_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - sparse_indices: (N, 4) sparse coordinates
                - sparse_features: (N, 1) wavelet coefficients
                - category: Category index
                - category_name: Category string
                - mesh_path: Path to original mesh
        """
        sample = self.samples[idx]
        mesh_path = sample['path']
        category = sample['category']
        category_idx = sample['category_idx']
        
        # Check cache
        if self.cache_sdf:
            cache_file = self.cache_dir / f"{mesh_path.stem}_{self.resolution}.npz"
            if cache_file.exists():
                cached = np.load(cache_file)
                sparse_data = {
                    'indices': cached['indices'],
                    'features': cached['features'],
                    'shape': tuple(cached['shape']),
                    'wavelet': str(cached['wavelet']),
                    'level': int(cached['level'])
                }
            else:
                sparse_data = self._process_mesh(mesh_path)
                # Save to cache
                np.savez_compressed(
                    cache_file,
                    indices=sparse_data['indices'],
                    features=sparse_data['features'],
                    shape=np.array(sparse_data['shape']),
                    wavelet=sparse_data['wavelet'],
                    level=sparse_data['level']
                )
        else:
            sparse_data = self._process_mesh(mesh_path)
        
        # Convert to torch tensors
        result = {
            'sparse_indices': torch.from_numpy(sparse_data['indices']).long(),
            'sparse_features': torch.from_numpy(sparse_data['features']).float(),
            'category': torch.tensor(category_idx, dtype=torch.long),
            'category_name': category,
            'mesh_path': str(mesh_path),
            'shape': sparse_data['shape']
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def _process_mesh(self, mesh_path: Path) -> Dict:
        """Load mesh, convert to SDF, apply wavelet transform."""
        try:
            # Load and normalize mesh
            mesh = trimesh.load(mesh_path, force='mesh')
            mesh = normalize_mesh(mesh)
            
            # Convert to SDF
            sdf = mesh_to_sdf_simple(mesh, resolution=self.resolution)
            
            # Wavelet transform
            sparse_data = sdf_to_sparse_wavelet(
                sdf,
                threshold=self.wavelet_threshold,
                wavelet=self.wavelet_type,
                level=self.wavelet_level
            )
            
            return sparse_data
            
        except Exception as e:
            print(f"❌ Error processing {mesh_path}: {e}")
            # Return empty sparse data
            return {
                'indices': np.zeros((0, 4), dtype=np.int32),
                'features': np.zeros((0, 1), dtype=np.float32),
                'shape': (self.resolution,) * 3,
                'wavelet': self.wavelet_type,
                'level': self.wavelet_level
            }


class ShapeNetDataset(Dataset):
    """
    ShapeNet Core v2 dataset.
    
    Structure:
        data/ShapeNet/
            02691156/  # Category ID (airplane)
                1a04e3eab45ca15dd86060f189eb133/
                    models/
                        model_normalized.obj
            ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 32,
        wavelet_threshold: float = 0.01,
        wavelet_type: str = 'bior4.4',
        wavelet_level: int = 3,
        categories: Optional[List[str]] = None,
        split_file: Optional[str] = None,
        transform=None,
        cache_sdf: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            root_dir: Path to ShapeNet directory
            split: 'train', 'val', or 'test'
            split_file: JSON file with train/val/test splits
            Other args same as ModelNet40Dataset
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.resolution = resolution
        self.wavelet_threshold = wavelet_threshold
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.transform = transform
        self.cache_sdf = cache_sdf
        
        # Load split file
        if split_file and Path(split_file).exists():
            with open(split_file) as f:
                splits = json.load(f)
            split_ids = splits.get(split, [])
        else:
            # Auto-generate splits
            split_ids = None
        
        # Find all meshes
        self.samples = []
        all_categories = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        if categories is None:
            categories = all_categories
        
        for cat in categories:
            cat_path = self.root_dir / cat
            if not cat_path.exists():
                continue
            
            for model_dir in cat_path.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # Check if in split
                if split_ids is not None and model_dir.name not in split_ids:
                    continue
                
                # Find mesh file
                mesh_file = model_dir / "models" / "model_normalized.obj"
                if not mesh_file.exists():
                    # Try other formats
                    mesh_files = list((model_dir / "models").glob("model_normalized.*"))
                    if mesh_files:
                        mesh_file = mesh_files[0]
                    else:
                        continue
                
                self.samples.append({
                    'path': mesh_file,
                    'category': cat,
                    'category_idx': all_categories.index(cat),
                    'model_id': model_dir.name
                })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        self.num_categories = len(all_categories)
        self.categories = all_categories
        
        print(f"📊 ShapeNet {split}: {len(self.samples)} samples from {len(categories)} categories")
        
        # Cache directory
        if cache_sdf:
            self.cache_dir = self.root_dir / f"cache_sdf_{resolution}"
            self.cache_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Same as ModelNet40Dataset."""
        sample = self.samples[idx]
        mesh_path = sample['path']
        category = sample['category']
        category_idx = sample['category_idx']
        
        # Check cache
        if self.cache_sdf:
            cache_file = self.cache_dir / f"{sample['model_id']}_{self.resolution}.npz"
            if cache_file.exists():
                cached = np.load(cache_file)
                sparse_data = {
                    'indices': cached['indices'],
                    'features': cached['features'],
                    'shape': tuple(cached['shape']),
                    'wavelet': str(cached['wavelet']),
                    'level': int(cached['level'])
                }
            else:
                sparse_data = self._process_mesh(mesh_path)
                np.savez_compressed(
                    cache_file,
                    indices=sparse_data['indices'],
                    features=sparse_data['features'],
                    shape=np.array(sparse_data['shape']),
                    wavelet=sparse_data['wavelet'],
                    level=sparse_data['level']
                )
        else:
            sparse_data = self._process_mesh(mesh_path)
        
        result = {
            'sparse_indices': torch.from_numpy(sparse_data['indices']).long(),
            'sparse_features': torch.from_numpy(sparse_data['features']).float(),
            'category': torch.tensor(category_idx, dtype=torch.long),
            'category_name': category,
            'model_id': sample['model_id'],
            'mesh_path': str(mesh_path),
            'shape': sparse_data['shape']
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def _process_mesh(self, mesh_path: Path) -> Dict:
        """Same as ModelNet40Dataset."""
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            mesh = normalize_mesh(mesh)
            sdf = mesh_to_sdf_simple(mesh, resolution=self.resolution)
            sparse_data = sdf_to_sparse_wavelet(
                sdf,
                threshold=self.wavelet_threshold,
                wavelet=self.wavelet_type,
                level=self.wavelet_level
            )
            return sparse_data
        except Exception as e:
            print(f"❌ Error processing {mesh_path}: {e}")
            return {
                'indices': np.zeros((0, 4), dtype=np.int32),
                'features': np.zeros((0, 1), dtype=np.float32),
                'shape': (self.resolution,) * 3,
                'wavelet': self.wavelet_type,
                'level': self.wavelet_level
            }


def collate_sparse(batch: List[Dict]) -> Dict:
    """
    Custom collate function for sparse data.
    Batches sparse tensors by adding batch dimension to indices.
    """
    batch_size = len(batch)
    
    # Collect all sparse data
    all_indices = []
    all_features = []
    categories = []
    category_names = []
    mesh_paths = []
    shapes = []
    
    for i, sample in enumerate(batch):
        indices = sample['sparse_indices']
        features = sample['sparse_features']
        
        # Add batch dimension to indices
        batch_indices = torch.full((len(indices), 1), i, dtype=torch.long)
        indices_with_batch = torch.cat([batch_indices, indices], dim=1)
        
        all_indices.append(indices_with_batch)
        all_features.append(features)
        categories.append(sample['category'])
        category_names.append(sample['category_name'])
        mesh_paths.append(sample['mesh_path'])
        shapes.append(sample['shape'])
    
    # Concatenate
    batched = {
        'sparse_indices': torch.cat(all_indices, dim=0),  # (N_total, 5) - [batch, x, y, z, channel]
        'sparse_features': torch.cat(all_features, dim=0),  # (N_total, 1)
        'category': torch.stack(categories),  # (B,)
        'category_name': category_names,
        'mesh_path': mesh_paths,
        'shape': shapes,
        'batch_size': batch_size
    }
    
    return batched


# Helper function to create dataloaders
def create_dataloader(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    resolution: int = 32,
    **kwargs
):
    """
    Convenience function to create dataloader.
    
    Args:
        dataset_name: 'modelnet40' or 'shapenet'
        root_dir: Path to dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers
        resolution: SDF resolution
        **kwargs: Additional args for dataset
    """
    if dataset_name.lower() == 'modelnet40':
        dataset = ModelNet40Dataset(
            root_dir=root_dir,
            split=split,
            resolution=resolution,
            **kwargs
        )
    elif dataset_name.lower() == 'shapenet':
        dataset = ShapeNetDataset(
            root_dir=root_dir,
            split=split,
            resolution=resolution,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_sparse,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    print("Testing ModelNet40Dataset...")
    
    dataset = ModelNet40Dataset(
        root_dir="data/ModelNet40",
        split='train',
        resolution=32,
        max_samples=10,
        cache_sdf=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Categories: {dataset.categories[:5]}...")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Category: {sample['category_name']}")
    print(f"  Sparse indices: {sample['sparse_indices'].shape}")
    print(f"  Sparse features: {sample['sparse_features'].shape}")
    print(f"  Shape: {sample['shape']}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(
        dataset_name='modelnet40',
        root_dir='data/ModelNet40',
        split='train',
        batch_size=4,
        num_workers=0,
        resolution=32,
        max_samples=10
    )
    
    for batch in dataloader:
        print(f"Batch:")
        print(f"  Sparse indices: {batch['sparse_indices'].shape}")
        print(f"  Sparse features: {batch['sparse_features'].shape}")
        print(f"  Categories: {batch['category']}")
        print(f"  Category names: {batch['category_name']}")
        break
    
    print("\n✅ Dataset tests passed!")
