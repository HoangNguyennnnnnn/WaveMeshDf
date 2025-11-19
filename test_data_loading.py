"""
Test data loading and conversion to dense tensor
"""

import torch
from data.mesh_dataset import create_dataloader

print("=" * 60)
print("Data Loading Test")
print("=" * 60)

print("\n1. Creating dataloader...")
dataloader = create_dataloader(
    dataset_name='modelnet40',
    root_dir='data/ModelNet40',
    split='train',
    batch_size=2,
    num_workers=0,
    resolution=16,
    max_samples=5
)

print(f"   Dataset size: {len(dataloader.dataset)}")

print("\n2. Loading one batch...")
batch = next(iter(dataloader))

print(f"   Batch keys: {list(batch.keys())}")
print(f"   sparse_indices shape: {batch['sparse_indices'].shape}")
print(f"   sparse_features shape: {batch['sparse_features'].shape}")
print(f"   categories: {batch['category_name']}")
print(f"   batch_size: {batch['batch_size']}")

print("\n3. Converting sparse to dense...")
sparse_indices = batch['sparse_indices']
sparse_features = batch['sparse_features']
batch_size = batch['batch_size']
resolution = 16

print(f"   Indices format: [batch, x, y, z, channel]")
print(f"   Sample indices (first 5):")
print(f"   {sparse_indices[:5]}")

# Convert to dense
x = torch.zeros(batch_size, 1, resolution, resolution, resolution)

for b in range(batch_size):
    batch_mask = (sparse_indices[:, 0] == b)
    n_points = batch_mask.sum().item()
    print(f"\n   Batch {b}: {n_points} non-zero coefficients")
    
    if batch_mask.any():
        indices = sparse_indices[batch_mask][:, 1:4].long()
        features = sparse_features[batch_mask].squeeze(-1)
        
        # Aggregate
        for i in range(len(indices)):
            x[b, 0, indices[i, 0], indices[i, 1], indices[i, 2]] += features[i]

print(f"\n4. Dense tensor created:")
print(f"   Shape: {x.shape}")
print(f"   Non-zero elements: {(x != 0).sum().item()} / {x.numel()}")
print(f"   Sparsity: {100 * (1 - (x != 0).sum().item() / x.numel()):.1f}%")
print(f"   Value range: [{x.min():.3f}, {x.max():.3f}]")

print("\n✅ Data loading works!")
print("=" * 60)
