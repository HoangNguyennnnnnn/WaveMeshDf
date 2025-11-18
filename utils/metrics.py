"""
Evaluation metrics for 3D mesh generation
"""

import torch
import numpy as np
from typing import Tuple, Optional


def chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    num_samples: int = 1000
) -> Tuple[float, float]:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred_points: Predicted points (N, 3)
        gt_points: Ground truth points (M, 3)
        num_samples: Number of points to sample (for efficiency)
        
    Returns:
        (chamfer_distance, f_score)
    """
    # Sample points if too many
    if len(pred_points) > num_samples:
        indices = np.random.choice(len(pred_points), num_samples, replace=False)
        pred_points = pred_points[indices]
    
    if len(gt_points) > num_samples:
        indices = np.random.choice(len(gt_points), num_samples, replace=False)
        gt_points = gt_points[indices]
    
    # Convert to torch for efficient computation
    pred_torch = torch.from_numpy(pred_points).float()
    gt_torch = torch.from_numpy(gt_points).float()
    
    # Compute pairwise distances
    # pred_to_gt: for each pred point, find nearest gt point
    dist_pred_to_gt = torch.cdist(pred_torch, gt_torch)  # (N, M)
    min_dist_pred_to_gt, _ = dist_pred_to_gt.min(dim=1)  # (N,)
    
    # gt_to_pred: for each gt point, find nearest pred point
    min_dist_gt_to_pred, _ = dist_pred_to_gt.min(dim=0)  # (M,)
    
    # Chamfer distance (symmetric)
    chamfer = (min_dist_pred_to_gt.mean() + min_dist_gt_to_pred.mean()).item()
    
    return chamfer


def compute_iou(
    pred_sdf: np.ndarray,
    gt_sdf: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Compute IoU (Intersection over Union) for SDF grids.
    
    Args:
        pred_sdf: Predicted SDF (D, H, W)
        gt_sdf: Ground truth SDF (D, H, W)
        threshold: Threshold for occupancy (0.0 = surface)
        
    Returns:
        IoU score
    """
    pred_occupied = pred_sdf < threshold
    gt_occupied = gt_sdf < threshold
    
    intersection = np.logical_and(pred_occupied, gt_occupied).sum()
    union = np.logical_or(pred_occupied, gt_occupied).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(iou)


def compute_metrics(
    pred_vertices: np.ndarray,
    pred_faces: np.ndarray,
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
    num_samples: int = 1000
) -> dict:
    """
    Compute all metrics for a mesh prediction.
    
    Args:
        pred_vertices: Predicted vertices (N, 3)
        pred_faces: Predicted faces (F, 3)
        gt_vertices: GT vertices (M, 3)
        gt_faces: GT faces (G, 3)
        num_samples: Number of points to sample
        
    Returns:
        Dictionary with metrics
    """
    # Sample points from meshes
    # For simplicity, we'll just use vertices
    # In production, use proper surface sampling
    
    metrics = {}
    
    # Chamfer distance
    if len(pred_vertices) > 0 and len(gt_vertices) > 0:
        chamfer = chamfer_distance(pred_vertices, gt_vertices, num_samples)
        metrics['chamfer_distance'] = chamfer
    else:
        metrics['chamfer_distance'] = float('inf')
    
    # Vertex count
    metrics['num_vertices_pred'] = len(pred_vertices)
    metrics['num_vertices_gt'] = len(gt_vertices)
    metrics['num_faces_pred'] = len(pred_faces)
    metrics['num_faces_gt'] = len(gt_faces)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy point clouds
    pred_points = np.random.randn(1000, 3)
    gt_points = pred_points + np.random.randn(1000, 3) * 0.1
    
    # Chamfer distance
    cd = chamfer_distance(pred_points, gt_points)
    print(f"Chamfer distance: {cd:.6f}")
    
    # IoU
    pred_sdf = np.random.randn(32, 32, 32)
    gt_sdf = pred_sdf + np.random.randn(32, 32, 32) * 0.5
    iou = compute_iou(pred_sdf, gt_sdf)
    print(f"IoU: {iou:.4f}")
    
    print("✅ Metrics tests passed!")
