"""
Training Script for WaveMesh-Diff
Main training loop with logging, checkpointing, and evaluation
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

from data.mesh_dataset import create_dataloader
from models import WaveMeshUNet, GaussianDiffusion, create_multiview_encoder
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train WaveMesh-Diff')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'shapenet'])
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--resolution', type=int, default=32, help='SDF resolution')
    parser.add_argument('--wavelet_threshold', type=float, default=0.01)
    
    # Model
    parser.add_argument('--unet_channels', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--time_emb_dim', type=int, default=256)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--context_dim', type=int, default=384, help='Multi-view encoder output dim')
    
    # Diffusion
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_encoder', type=float, default=1e-5, help='LR for pretrained encoder')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Optimizer & Scheduler
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_freq', type=int, default=5, help='Save every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=100, help='Log every N steps')
    parser.add_argument('--val_freq', type=int, default=1, help='Validate every N epochs')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=0, help='Data loading workers (use 0 for Colab)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mixed_precision', action='store_true')
    
    # Advanced
    parser.add_argument('--use_ema', action='store_true', help='Use EMA for model weights')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--cfg_prob', type=float, default=0.1, help='Classifier-free guidance dropout')
    
    # Debug
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    return args


def create_model(args):
    """Initialize models."""
    # U-Net
    encoder_channels = args.unet_channels
    decoder_channels = list(reversed(args.unet_channels))
    
    unet = WaveMeshUNet(
        in_channels=1,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        time_emb_dim=args.time_emb_dim,
        use_attention=args.use_attention,
        context_dim=args.context_dim if args.context_dim > 0 else None
    )
    
    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.diffusion_steps,
        beta_schedule=args.beta_schedule
    )
    
    # Multi-view encoder (optional)
    encoder = None
    if args.context_dim > 0:
        encoder = create_multiview_encoder(
            preset='small',
            feature_dim=args.context_dim
        )
    
    return unet, diffusion, encoder


def create_optimizer(args, unet, encoder=None):
    """Create optimizer with separate learning rates."""
    params = [
        {'params': unet.parameters(), 'lr': args.lr}
    ]
    
    if encoder is not None:
        params.append({
            'params': encoder.parameters(),
            'lr': args.lr_encoder
        })
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def create_scheduler(args, optimizer, num_training_steps):
    """Create learning rate scheduler."""
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_training_steps // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(args, unet, diffusion, encoder, train_loader, optimizer, scheduler, scaler, epoch, logger):
    """Train for one epoch."""
    unet.train()
    if encoder is not None:
        encoder.train()
    
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        # Get data
        sparse_indices = batch['sparse_indices'].to(args.device)  # (N, 5): [batch, x, y, z, channel]
        sparse_features = batch['sparse_features'].to(args.device)  # (N, 1)
        batch_size = len(batch['category'])
        
        # Convert sparse to dense tensor for training
        # TODO: Optimize with spconv later for better performance
        # For now, aggregate all channels into single feature map
        x = torch.zeros(
            batch_size, 1, 
            args.resolution, args.resolution, args.resolution
        ).to(args.device)
        
        # Fill in sparse values (aggregate multi-channel coefficients)
        for b in range(batch_size):
            # Get indices for this batch item
            batch_mask = (sparse_indices[:, 0] == b)
            if batch_mask.any():
                indices = sparse_indices[batch_mask][:, 1:4].long()  # [x, y, z] only (drop channel)
                features = sparse_features[batch_mask].squeeze(-1)  # (N,)
                
                # Clamp indices to valid range
                max_val = args.resolution - 1
                indices = torch.clamp(indices, 0, max_val)
                
                # Aggregate features at same spatial location (sum over channels)
                for i in range(len(indices)):
                    x[b, 0, indices[i, 0], indices[i, 1], indices[i, 2]] += features[i]
        
        # Get conditioning (if using encoder)
        context = None
        if encoder is not None:
            # TODO: Need multi-view images here
            # For now, use None (unconditional training)
            pass
        
        # Random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=args.device)
        
        # Forward diffusion (add noise)
        noise = torch.randn_like(x)
        x_noisy = diffusion.q_sample(x, t, noise=noise)
        
        # Predict noise with U-Net
        with torch.amp.autocast('cuda', enabled=args.mixed_precision):
            noise_pred = unet(x_noisy, t, context=context)
            
            # Loss: MSE between predicted and actual noise
            loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        if args.mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            if encoder is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            if encoder is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if (step + 1) % args.log_freq == 0:
            logger.log({
                'train/loss': loss.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'step': step
            })
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(args, unet, diffusion, encoder, val_loader, epoch, logger):
    """Validation loop."""
    unet.eval()
    if encoder is not None:
        encoder.eval()
    
    total_loss = 0
    num_batches = len(val_loader)
    
    for batch in tqdm(val_loader, desc="Validating"):
        batch_size = batch['batch_size']
        x = torch.randn(batch_size, 1, args.resolution, args.resolution, args.resolution).to(args.device)
        
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=args.device)
        noise = torch.randn_like(x)
        x_noisy = diffusion.q_sample(x, t, noise=noise)
        
        noise_pred = unet(x_noisy, t, context=None)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.log({
        'val/loss': avg_loss,
        'epoch': epoch
    })
    
    return avg_loss


def main():
    args = parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_res{args.resolution}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info(f"Starting training: {exp_name}")
    logger.info(f"Arguments: {args}")
    
    # Create dataloaders
    logger.info("Creating datasets...")
    
    # Auto-detect memory constraints (Colab has ~12GB RAM)
    if args.num_workers > 0:
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            if available_ram_gb < 8:
                logger.warning(f"Low RAM detected ({available_ram_gb:.1f}GB). Setting num_workers=0 to avoid OOM.")
                args.num_workers = 0
        except ImportError:
            pass
    
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        wavelet_threshold=args.wavelet_threshold,
        cache_sdf=True,
        max_samples=args.max_samples if args.debug else None
    )
    
    val_loader = create_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        wavelet_threshold=args.wavelet_threshold,
        cache_sdf=True,
        max_samples=100 if args.debug else None
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create models
    logger.info("Creating models...")
    unet, diffusion, encoder = create_model(args)
    unet = unet.to(args.device)
    diffusion = diffusion.to(args.device)
    if encoder is not None:
        encoder = encoder.to(args.device)
    
    total_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"U-Net parameters: {total_params:,}")
    if encoder is not None:
        encoder_params = sum(p.numel() for p in encoder.parameters())
        logger.info(f"Encoder parameters: {encoder_params:,}")
    
    # Create optimizer
    optimizer = create_optimizer(args, unet, encoder)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = create_scheduler(args, optimizer, num_training_steps)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        unet.load_state_dict(checkpoint['unet'])
        if encoder is not None and 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            args, unet, diffusion, encoder,
            train_loader, optimizer, scheduler, scaler,
            epoch, logger
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_loss = validate(args, unet, diffusion, encoder, val_loader, epoch, logger)
            logger.info(f"Val loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / 'best.pth',
                    unet=unet,
                    encoder=encoder,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss
                )
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                output_dir / f'checkpoint_epoch_{epoch}.pth',
                unet=unet,
                encoder=encoder,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss
            )
    
    # Save final model
    save_checkpoint(
        output_dir / 'final.pth',
        unet=unet,
        encoder=encoder,
        optimizer=optimizer,
        epoch=args.epochs - 1,
        loss=train_loss
    )
    
    logger.info("Training complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
