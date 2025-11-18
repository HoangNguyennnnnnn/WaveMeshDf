"""
Checkpoint utilities for saving/loading models
"""

import torch
from pathlib import Path
from typing import Optional, Dict


def save_checkpoint(
    path: str,
    unet: torch.nn.Module,
    encoder: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    **kwargs
):
    """
    Save checkpoint to disk.
    
    Args:
        path: Path to save checkpoint
        unet: U-Net model
        encoder: Optional encoder model
        optimizer: Optional optimizer
        epoch: Current epoch
        loss: Current loss
        **kwargs: Additional metadata
    """
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'unet': unet.state_dict(),
    }
    
    if encoder is not None:
        checkpoint['encoder'] = encoder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    # Add additional metadata
    checkpoint.update(kwargs)
    
    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: str = 'cpu'
) -> Dict:
    """
    Load checkpoint from disk.
    
    Args:
        path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    key: str = 'unet',
    strict: bool = True
):
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        key: Key in checkpoint dict
        strict: Whether to strictly enforce matching keys
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if key in checkpoint:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
