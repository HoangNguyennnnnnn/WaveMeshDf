"""
Diffusion Model for WaveMesh-Diff
Implements DDPM and DDIM for sparse wavelet coefficient generation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for sparse wavelet coefficients.
    Implements both DDPM (training) and DDIM (fast sampling).
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_denoised: bool = True,
        predict_epsilon: bool = True
    ):
        """
        Args:
            timesteps: Number of diffusion steps
            beta_schedule: 'linear', 'cosine', or 'sqrt'
            beta_start: Starting beta value
            beta_end: Ending beta value
            clip_denoised: Whether to clip predicted values
            predict_epsilon: If True, predict noise; else predict x0
        """
        super().__init__()
        
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        
        # Create beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sqrt':
            betas = self._sqrt_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.register_buffer('betas', betas)
        
        # Pre-compute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sqrt_beta_schedule(self, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        """Square root schedule"""
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to x_start according to timestep t.
        
        Args:
            x_start: Clean data (batch, ...)
            t: Timestep (batch,)
            noise: Noise to add (same shape as x_start)
        
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Broadcast to match input shape
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and noise."""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Args:
            model_output: Output from denoising model
            x_t: Current noisy sample
            t: Current timestep
        
        Returns:
            Mean, variance, and log variance
        """
        if self.predict_epsilon:
            # Model predicts noise
            pred_x_start = self.predict_start_from_noise(x_t, t, model_output)
        else:
            # Model predicts x_0 directly
            pred_x_start = model_output
        
        if self.clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(pred_x_start, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) using model output.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(model_output, x_t, t)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        sparse_indices: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        progress: bool = True
    ) -> torch.Tensor:
        """
        DDPM sampling loop.
        
        Args:
            model: Denoising model
            shape: Shape of output features
            sparse_indices: Sparse tensor indices (fixed during sampling)
            context: Conditioning information
            progress: Show progress bar
        
        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        timesteps = list(range(self.timesteps))[::-1]
        
        if progress:
            timesteps = tqdm(timesteps, desc='DDPM Sampling')
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Create sparse tensor
            import spconv.pytorch as spconv
            x_sparse = spconv.SparseConvTensor(
                features=x,
                indices=sparse_indices,
                spatial_shape=[64, 64, 64],  # TODO: make this configurable
                batch_size=shape[0]
            )
            
            # Model prediction
            with torch.no_grad():
                model_output = model(x_sparse, t_batch, context)
                model_output_features = model_output.features
            
            # Sample x_{t-1}
            x = self.p_sample(model_output_features, x, t_batch)
        
        return x
    
    def ddim_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM sampling step (deterministic when eta=0).
        
        Args:
            model_output: Model prediction
            x_t: Current sample
            t: Current timestep
            t_prev: Previous timestep
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        
        Returns:
            x_{t_prev}
        """
        if self.predict_epsilon:
            pred_epsilon = model_output
            pred_x_start = self.predict_start_from_noise(x_t, t, pred_epsilon)
        else:
            pred_x_start = model_output
            pred_epsilon = (
                (x_t - self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * pred_x_start) /
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            )
        
        if self.clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # Compute variance
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        
        sigma = (
            eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) *
            torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )
        
        # Compute mean
        pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * pred_epsilon
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x_start + pred_dir
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
        
        return x_prev
    
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        sparse_indices: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        DDIM sampling loop (faster than DDPM).
        
        Args:
            model: Denoising model
            shape: Shape of output features
            sparse_indices: Sparse tensor indices
            context: Conditioning information
            ddim_steps: Number of sampling steps (< timesteps)
            eta: Stochasticity (0=deterministic)
            progress: Show progress bar
        
        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        
        # Create time steps
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]
        timesteps_prev = [0] + timesteps[:-1]
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        if progress:
            pbar = tqdm(zip(timesteps, timesteps_prev), total=len(timesteps), desc='DDIM Sampling')
        else:
            pbar = zip(timesteps, timesteps_prev)
        
        for t, t_prev in pbar:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            
            # Create sparse tensor
            import spconv.pytorch as spconv
            x_sparse = spconv.SparseConvTensor(
                features=x,
                indices=sparse_indices,
                spatial_shape=[64, 64, 64],  # TODO: configurable
                batch_size=shape[0]
            )
            
            # Model prediction
            with torch.no_grad():
                model_output = model(x_sparse, t_batch, context)
                model_output_features = model_output.features
            
            # DDIM step
            x = self.ddim_sample(model_output_features, x, t_batch, t_prev_batch, eta)
        
        return x
    
    def training_losses(
        self,
        model: nn.Module,
        x_start_sparse: 'spconv.SparseConvTensor',
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            model: Denoising model
            x_start_sparse: Clean sparse tensor
            t: Timesteps
            context: Conditioning
            noise: Noise to add (optional)
        
        Returns:
            Dictionary of losses
        """
        x_start = x_start_sparse.features
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise
        x_t = self.q_sample(x_start, t, noise)
        
        # Create noisy sparse tensor
        x_t_sparse = x_start_sparse.replace_feature(x_t)
        
        # Model prediction
        model_output = model(x_t_sparse, t, context)
        model_output_features = model_output.features
        
        # Compute loss
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start
        
        # MSE loss
        mse_loss = torch.nn.functional.mse_loss(model_output_features, target)
        
        return {
            'loss': mse_loss,
            'mse': mse_loss
        }
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps and reshape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


if __name__ == "__main__":
    # Quick test
    print("Testing GaussianDiffusion...")
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_schedule='linear',
        predict_epsilon=True
    )
    
    # Test forward process
    x_start = torch.randn(4, 100, 1)  # (batch, num_points, features)
    t = torch.randint(0, 1000, (4,))
    
    x_t = diffusion.q_sample(x_start, t)
    print(f"✓ Forward diffusion: {x_start.shape} -> {x_t.shape}")
    
    # Test loss computation would require a model
    print("✓ GaussianDiffusion initialized successfully!")
    print(f"  - Timesteps: {diffusion.timesteps}")
    print(f"  - Beta range: [{diffusion.betas[0]:.6f}, {diffusion.betas[-1]:.6f}]")
