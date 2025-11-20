"""
Sparse 3D U-Net for WaveMesh-Diff
Uses sparse convolutions to efficiently process sparse wavelet coefficients
Compatible with Google Colab (uses dense fallback when spconv unavailable)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

# Use compatibility layer for spconv
from .spconv_compat import (
    SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor, 
    SparseSequential, BatchNorm, is_spconv_available
)


class SparseResBlock(nn.Module):
    """
    Sparse Residual Block with submanifold convolutions.
    Maintains sparsity pattern while learning features.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_subm: bool = True,
        indice_key: Optional[str] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # First convolution
        if use_subm and stride == 1:
            self.conv1 = SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key
            )
        else:
            self.conv1 = SparseConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=indice_key
            )
        
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution (always submanifold)
        self.conv2 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key
        )
        self.bn2 = BatchNorm(out_channels)
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = SparseSequential(
                SparseConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                BatchNorm(out_channels)
            )
        else:
            self.shortcut = None
    
    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x
        
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        # Add residual
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        
        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block to inject image conditioning into sparse features.
    Adapts standard multi-head attention for sparse tensors.
    """
    
    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (from sparse features)
        self.to_q = nn.Linear(feature_dim, feature_dim)
        
        # Key and Value projections (from image context)
        self.to_k = nn.Linear(context_dim, feature_dim)
        self.to_v = nn.Linear(context_dim, feature_dim)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        x: SparseConvTensor,
        context: torch.Tensor
    ) -> SparseConvTensor:
        """
        Args:
            x: Sparse features (N_sparse, feature_dim)
            context: Image features (batch, num_views, context_dim)
        
        Returns:
            Attended sparse features
        """
        batch_size = x.batch_size
        features = x.features  # (N_sparse, feature_dim)
        
        # Get batch indices for each sparse feature
        batch_indices = x.indices[:, 0]  # (N_sparse,)
        
        # Normalize
        features = self.norm(features)
        
        # Project to Q, K, V
        q = self.to_q(features)  # (N_sparse, feature_dim)
        
        # Flatten context for attention
        context_flat = context.reshape(batch_size, -1, self.context_dim)  # (batch, num_views, context_dim)
        k = self.to_k(context_flat)  # (batch, num_views, feature_dim)
        v = self.to_v(context_flat)  # (batch, num_views, feature_dim)
        
        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)  # (N_sparse, heads, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)  # (batch, num_views, heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)  # (batch, num_views, heads, head_dim)
        
        # Compute attention for each batch
        attended_features = []
        for b in range(batch_size):
            # Get queries for this batch
            mask_b = batch_indices == b
            q_b = q[mask_b]  # (N_b, heads, head_dim)
            
            if q_b.shape[0] == 0:
                continue
            
            # Get keys and values for this batch
            k_b = k[b]  # (num_views, heads, head_dim)
            v_b = v[b]  # (num_views, heads, head_dim)
            
            # Compute attention scores
            # q_b: (N_b, heads, head_dim) @ k_b.T: (heads, head_dim, num_views)
            # -> (N_b, heads, num_views)
            attn = torch.einsum('nhd,mhd->nhm', q_b, k_b) * self.scale
            attn = torch.softmax(attn, dim=-1)
            
            # Apply attention to values
            # attn: (N_b, heads, num_views) @ v_b: (num_views, heads, head_dim)
            # -> (N_b, heads, head_dim)
            out_b = torch.einsum('nhm,mhd->nhd', attn, v_b)
            out_b = out_b.reshape(out_b.shape[0], -1)  # (N_b, feature_dim)
            
            attended_features.append(out_b)
        
        # Concatenate all batches
        if len(attended_features) > 0:
            attended = torch.cat(attended_features, dim=0)  # (N_sparse, feature_dim)
        else:
            attended = torch.zeros_like(features)
        
        # Output projection
        out = self.to_out(attended)
        
        # Residual connection
        out = features + out
        
        return x.replace_feature(out)


class SparseUNetEncoder(nn.Module):
    """Encoder part of Sparse U-Net with downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [32, 64, 128, 256],
        strides: List[int] = [1, 2, 2, 2]
    ):
        super().__init__()
        
        self.channels = channels
        self.strides = strides
        
        # Input convolution
        self.input_conv = SparseSequential(
            SubMConv3d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(
                SparseResBlock(
                    channels[i],
                    channels[i + 1],
                    stride=strides[i + 1],
                    use_subm=(strides[i + 1] == 1),
                    indice_key=f'enc_{i}'
                )
            )
    
    def forward(self, x: SparseConvTensor) -> Tuple[SparseConvTensor, List[SparseConvTensor]]:
        x = self.input_conv(x)
        
        skip_connections = []
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x)
        
        return x, skip_connections


class SparseUNetDecoder(nn.Module):
    """Decoder part of Sparse U-Net with upsampling."""
    
    def __init__(
        self,
        channels: List[int] = [256, 128, 64, 32],
        use_attention: bool = True,
        context_dim: int = 768,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.channels = channels
        self.use_attention = use_attention
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            # Upsampling
            self.upsample_blocks.append(
                SparseInverseConv3d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    bias=False,
                    indice_key=f'enc_{len(channels) - 2 - i}'  # Match encoder keys
                )
            )
            
            # Decoder block (with skip connection, so double channels)
            self.decoder_blocks.append(
                SparseResBlock(
                    channels[i + 1] * 2,  # Concatenated with skip
                    channels[i + 1],
                    stride=1,
                    use_subm=True,
                    indice_key=f'dec_{i}'
                )
            )
        
        # Cross-attention blocks (one per decoder level)
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                CrossAttentionBlock(channels[i + 1], context_dim, num_heads)
                for i in range(len(channels) - 1)
            ])
    
    def forward(
        self,
        x: SparseConvTensor,
        skip_connections: List[SparseConvTensor],
        context: Optional[torch.Tensor] = None
    ) -> SparseConvTensor:
        
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            # Upsample
            x = upsample(x)
            
            # Concatenate with skip connection
            skip = skip_connections[-(i + 1)]
            
            # Handle potential index mismatch in dense fallback mode
            if hasattr(x, 'sample_at_indices') and len(x.indices) != len(skip.indices):
                # Align upsampled tensor to skip connection indices
                x = x.sample_at_indices(skip.indices)
            
            x = x.replace_feature(torch.cat([x.features, skip.features], dim=1))
            
            # Decoder block
            x = decoder_block(x)
            
            # Cross-attention with image features
            if self.use_attention and context is not None:
                x = self.attention_blocks[i](x, context)
        
        return x


class WaveMeshUNet(nn.Module):
    """
    Complete Sparse 3D U-Net for wavelet coefficient denoising.
    Takes noisy sparse wavelet coefficients and predicts noise/clean coefficients.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 128, 256],
        encoder_strides: List[int] = [1, 2, 2, 2],
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_attention: bool = True,
        context_dim: int = 768,
        num_heads: int = 8,
        time_emb_dim: int = 128
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding (for diffusion timestep)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, encoder_channels[0]),
            nn.SiLU(),
            nn.Linear(encoder_channels[0], encoder_channels[0])
        )
        
        # Encoder
        self.encoder = SparseUNetEncoder(
            in_channels + encoder_channels[0],  # + encoder_channels[0] for time embedding
            encoder_channels,
            encoder_strides
        )
        
        # Decoder
        self.decoder = SparseUNetDecoder(
            decoder_channels,
            use_attention,
            context_dim,
            num_heads
        )
        
        # Output head
        self.output_conv = SubMConv3d(
            decoder_channels[-1],
            out_channels,
            kernel_size=1,
            bias=True
        )
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Sinusoidal timestep embeddings (similar to Transformer positional encoding).
        
        Args:
            timesteps: (batch_size,) tensor of timesteps
            dim: Embedding dimension
        
        Returns:
            (batch_size, dim) embeddings
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if dim % 2 == 1:  # Odd dimension
            emb = torch.nn.functional.pad(emb, (0, 1))
        
        return emb
    
    def forward(
        self,
        x: Union[SparseConvTensor, torch.Tensor],
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Union[SparseConvTensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Sparse tensor with noisy wavelet coefficients OR dense tensor (B, C, D, H, W)
            timesteps: (batch_size,) diffusion timesteps
            context: (batch_size, num_views, context_dim) image features (optional)
        
        Returns:
            Sparse tensor with predicted noise/coefficients OR dense tensor (same format as input)
        """
        # Handle dense tensor input (fallback mode when spconv not available)
        is_dense = isinstance(x, torch.Tensor)
        if is_dense:
            # Use dense fallback implementation
            return self._forward_dense(x, timesteps, context)
        
        batch_size = x.batch_size
        
        # Get time embeddings
        t_emb = self.get_timestep_embedding(timesteps, self.time_emb_dim)  # (batch, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (batch, encoder_channels[0])
        
        # Broadcast time embedding to all sparse features
        batch_indices = x.indices[:, 0]
        t_features = t_emb[batch_indices]  # (N_sparse, encoder_channels[0])
        
        # Concatenate time features with spatial features
        x = x.replace_feature(torch.cat([x.features, t_features], dim=1))
        
        # Encode
        x, skip_connections = self.encoder(x)
        
        # Decode with cross-attention
        x = self.decoder(x, skip_connections, context)
        
        # Output
        x = self.output_conv(x)
        
        return x
    
    def _forward_dense(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dense fallback forward pass when spconv is not available.
        Uses standard 3D convolutions in a simple U-Net architecture.
        
        Args:
            x: Dense tensor (B, C, D, H, W)
            timesteps: (batch_size,) diffusion timesteps
            context: (batch_size, num_views, context_dim) image features (optional)
        
        Returns:
            Dense tensor (B, C, D, H, W) with predicted noise
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]  # (D, H, W)
        
        # Get time embeddings
        t_emb = self.get_timestep_embedding(timesteps, self.time_emb_dim)  # (batch, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (batch, encoder_channels[0])
        
        # Broadcast time embedding to spatial dimensions
        t_emb = t_emb.view(batch_size, -1, 1, 1, 1)  # (B, C, 1, 1, 1)
        t_emb = t_emb.expand(-1, -1, *spatial_shape)  # (B, C, D, H, W)
        
        # Create simple dense U-Net on the fly
        # Note: This is less efficient but works without spconv
        in_channels = x.shape[1]
        
        # Encoder
        # Down 1: in_channels -> encoder_channels[0]
        if not hasattr(self, 'dense_down1'):
            self.dense_down1 = nn.Sequential(
                nn.Conv3d(in_channels + self.encoder_channels[0], self.encoder_channels[0], 3, padding=1),
                nn.GroupNorm(8, self.encoder_channels[0]),
                nn.SiLU(),
                nn.Conv3d(self.encoder_channels[0], self.encoder_channels[0], 3, padding=1),
                nn.GroupNorm(8, self.encoder_channels[0]),
                nn.SiLU()
            ).to(x.device)
        
        # Down 2: encoder_channels[0] -> encoder_channels[1]
        if not hasattr(self, 'dense_down2'):
            self.dense_down2 = nn.Sequential(
                nn.Conv3d(self.encoder_channels[0], self.encoder_channels[1], 3, stride=2, padding=1),
                nn.GroupNorm(8, self.encoder_channels[1]),
                nn.SiLU(),
                nn.Conv3d(self.encoder_channels[1], self.encoder_channels[1], 3, padding=1),
                nn.GroupNorm(8, self.encoder_channels[1]),
                nn.SiLU()
            ).to(x.device)
        
        # Up 1: encoder_channels[1] -> encoder_channels[0]
        if not hasattr(self, 'dense_up1'):
            self.dense_up1 = nn.Sequential(
                nn.ConvTranspose3d(self.encoder_channels[1], self.encoder_channels[0], 2, stride=2),
                nn.GroupNorm(8, self.encoder_channels[0]),
                nn.SiLU()
            ).to(x.device)
        
        # Output: encoder_channels[0] * 2 (skip) -> in_channels
        if not hasattr(self, 'dense_out'):
            self.dense_out = nn.Conv3d(self.encoder_channels[0] * 2, in_channels, 3, padding=1).to(x.device)
        
        # Forward pass
        x_in = torch.cat([x, t_emb], dim=1)
        
        # Encoder
        h1 = self.dense_down1(x_in)  # Same spatial size
        h2 = self.dense_down2(h1)    # Halved spatial size
        
        # Decoder with skip connections
        h_up = self.dense_up1(h2)    # Back to original spatial size
        h_cat = torch.cat([h_up, h1], dim=1)  # Skip connection
        out = self.dense_out(h_cat)
        
        return out


def create_sparse_tensor_from_wavelet(
    sparse_data: dict,
    batch_size: int = 1,
    device: str = 'cuda'
) -> SparseConvTensor:
    """
    Helper function to create SparseTensor from wavelet sparse representation.
    
    Args:
        sparse_data: Output from WaveletTransform3D.dense_to_sparse_wavelet
        batch_size: Batch size
        device: Device to place tensor on
    
    Returns:
        SparseConvTensor ready for neural network
    """
    indices = sparse_data['indices']  # (N, 4) - [x, y, z, channel]
    features = sparse_data['features']  # (N, 1)
    shape = sparse_data['shape']
    
    # Convert to torch if needed
    if not torch.is_tensor(indices):
        indices = torch.from_numpy(indices)
    if not torch.is_tensor(features):
        features = torch.from_numpy(features)
    
    # Add batch dimension to indices if needed
    if indices.shape[1] == 4:
        # Remove channel dimension, add batch dimension
        spatial_indices = indices[:, :3]  # (N, 3)
        batch_indices = torch.zeros((spatial_indices.shape[0], 1), dtype=torch.int32)
        indices = torch.cat([batch_indices, spatial_indices], dim=1)  # (N, 4) - [batch, x, y, z]
    
    indices = indices.int().to(device)
    features = features.float().to(device)
    
    # Create sparse tensor
    sparse_tensor = SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=list(shape),
        batch_size=batch_size
    )
    
    return sparse_tensor


if __name__ == "__main__":
    # Quick test
    print("Testing WaveMeshUNet...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy sparse tensor
    batch_size = 2
    num_points = 1000
    spatial_shape = [64, 64, 64]
    
    # Random sparse indices
    indices = torch.randint(0, 64, (num_points, 3))
    batch_indices = torch.randint(0, batch_size, (num_points, 1))
    indices = torch.cat([batch_indices, indices], dim=1).int().to(device)
    
    # Random features
    features = torch.randn(num_points, 1).to(device)
    
    # Create sparse tensor
    x = SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )
    
    # Create model
    model = WaveMeshUNet(
        in_channels=1,
        out_channels=1,
        encoder_channels=[16, 32, 64],
        encoder_strides=[1, 2, 2],
        decoder_channels=[64, 32, 16],
        use_attention=True,
        context_dim=768,
        num_heads=4
    ).to(device)
    
    # Random timesteps and context
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    context = torch.randn(batch_size, 4, 768).to(device)  # 4 views
    
    # Forward pass
    print(f"Input: {x.features.shape}, {x.indices.shape}")
    out = model(x, timesteps, context)
    print(f"Output: {out.features.shape}, {out.indices.shape}")
    print("✓ WaveMeshUNet test passed!")
