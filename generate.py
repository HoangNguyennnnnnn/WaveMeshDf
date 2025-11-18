"""
Inference script for WaveMesh-Diff
Generate 3D meshes from trained model
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm

from models import WaveMeshUNet, GaussianDiffusion
from data.wavelet_utils import sparse_wavelet_to_sdf, sdf_to_mesh
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Generate meshes with WaveMesh-Diff')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--resolution', type=int, default=32)
    
    # Generation
    parser.add_argument('--num_samples', type=int, default=10, help='Number of meshes to generate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=1000, help='Diffusion sampling steps')
    parser.add_argument('--output_dir', type=str, default='generated_meshes')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def sample_meshes(
    unet: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch_size: int,
    resolution: int,
    num_steps: int,
    device: str,
    context=None
):
    """
    Sample meshes using DDPM/DDIM.
    
    Args:
        unet: Trained U-Net
        diffusion: Diffusion model
        batch_size: Batch size
        resolution: SDF resolution
        num_steps: Number of denoising steps
        device: Device
        context: Optional conditioning
        
    Returns:
        Batch of denoised SDF grids
    """
    unet.eval()
    
    # Start from pure noise
    x = torch.randn(batch_size, 1, resolution, resolution, resolution).to(device)
    
    # Denoise iteratively
    timesteps = torch.linspace(num_steps - 1, 0, num_steps).long().to(device)
    
    for t in tqdm(timesteps, desc="Sampling"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = unet(x, t_batch, context=context)
        
        # Denoise one step (simplified DDPM)
        if t > 0:
            # Get beta and alpha values
            beta_t = diffusion.betas[t]
            alpha_t = diffusion.alphas[t]
            alpha_cumprod_t = diffusion.alphas_cumprod[t]
            alpha_cumprod_prev = diffusion.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0).to(device)
            
            # Compute x_{t-1}
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # Clip for stability
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            # Compute mean
            mean = (
                torch.sqrt(alpha_cumprod_prev) * diffusion.betas[t] / (1 - alpha_cumprod_t) * pred_x0 +
                torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x
            )
            
            # Add noise
            if t > 0:
                noise = torch.randn_like(x)
                variance = diffusion.betas[t]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        else:
            # Final step - no noise
            alpha_cumprod_t = diffusion.alphas_cumprod[t]
            x = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
    
    return x


def sdf_to_mesh_simple(sdf: np.ndarray) -> trimesh.Trimesh:
    """
    Convert SDF to mesh using marching cubes.
    
    Args:
        sdf: SDF grid (D, H, W)
        
    Returns:
        Trimesh object
    """
    vertices, faces = sdf_to_mesh(sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    
    # Create models
    # Note: We need to know model architecture from checkpoint or config
    # For simplicity, use default architecture
    print("Creating models...")
    unet = WaveMeshUNet(
        in_channels=1,
        encoder_channels=[16, 32, 64, 128],
        decoder_channels=[128, 64, 32, 16],
        time_emb_dim=256,
        use_attention=True,
        context_dim=None  # Unconditional for now
    )
    unet.load_state_dict(checkpoint['unet'])
    unet = unet.to(args.device)
    unet.eval()
    
    diffusion = GaussianDiffusion(timesteps=args.num_steps)
    diffusion = diffusion.to(args.device)
    
    print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Generate meshes
    print(f"\nGenerating {args.num_samples} meshes...")
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    mesh_idx = 0
    for batch_idx in range(num_batches):
        current_batch_size = min(args.batch_size, args.num_samples - mesh_idx)
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}")
        
        # Sample SDFs
        sdf_batch = sample_meshes(
            unet=unet,
            diffusion=diffusion,
            batch_size=current_batch_size,
            resolution=args.resolution,
            num_steps=args.num_steps,
            device=args.device,
            context=None
        )
        
        # Convert to meshes
        sdf_batch = sdf_batch.cpu().numpy()
        
        for i in range(current_batch_size):
            sdf = sdf_batch[i, 0]  # (D, H, W)
            
            try:
                # Convert to mesh
                mesh = sdf_to_mesh_simple(sdf)
                
                # Save mesh
                mesh_path = output_dir / f"mesh_{mesh_idx:04d}.obj"
                mesh.export(mesh_path)
                
                print(f"  Saved {mesh_path.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
            except Exception as e:
                print(f"  ❌ Failed to convert mesh {mesh_idx}: {e}")
            
            mesh_idx += 1
    
    print(f"\n✅ Generated {mesh_idx} meshes in {output_dir}")


if __name__ == "__main__":
    main()
