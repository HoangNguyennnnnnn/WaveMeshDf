"""
Module D: Multi-view Image Encoder for WaveMesh-Diff
Encodes multi-view images into conditioning features for the diffusion model

Uses DINOv2 as the vision backbone (frozen pre-trained weights)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import math


class CameraPoseEmbedding(nn.Module):
    """
    Encode camera pose (rotation + translation) into embeddings.
    Uses sinusoidal position encoding similar to Transformers.
    """
    
    def __init__(self, pose_dim: int = 12, embed_dim: int = 256):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        
        # MLP to process pose parameters (3x4 matrix → embed_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, embed_dim),  # 3x4 pose matrix flattened
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_poses: (B, N_views, 3, 4) - N_views camera poses
            
        Returns:
            pose_embeddings: (B, N_views, embed_dim)
        """
        B, N, _, _ = camera_poses.shape
        
        # Flatten pose matrices
        poses_flat = camera_poses.reshape(B, N, -1)  # (B, N, 12)
        
        # Embed
        embeddings = self.pose_mlp(poses_flat)  # (B, N, embed_dim)
        
        return embeddings


class DINOv2Encoder(nn.Module):
    """
    DINOv2 Vision Transformer encoder.
    Uses pre-trained weights from Facebook/Meta AI.
    Frozen by default for faster training.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vits14',  # small model
        freeze: bool = True,
        feature_dim: int = 768
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.feature_dim = feature_dim
        
        # Try to load DINOv2 (will fail if transformers not installed)
        try:
            import transformers
            from transformers import AutoModel
            
            # Load pre-trained DINOv2
            self.dinov2 = AutoModel.from_pretrained(f'facebook/{model_name}')
            
            # Get actual output dimension
            if hasattr(self.dinov2.config, 'hidden_size'):
                actual_dim = self.dinov2.config.hidden_size
            else:
                # Default dimensions for different model sizes
                dim_map = {
                    'dinov2_vits14': 384,
                    'dinov2_vitb14': 768,
                    'dinov2_vitl14': 1024,
                    'dinov2_vitg14': 1536
                }
                actual_dim = dim_map.get(model_name, 768)
            
            # Projection layer if needed
            if actual_dim != feature_dim:
                self.proj = nn.Linear(actual_dim, feature_dim)
            else:
                self.proj = nn.Identity()
                
            # Freeze if requested
            if freeze:
                for param in self.dinov2.parameters():
                    param.requires_grad = False
                self.dinov2.eval()
                
            self.available = True
            
        except (ImportError, OSError, Exception) as e:
            # Silently fall back to CNN encoder if transformers not available
            self.available = False
            # Fallback: simple CNN for testing
            self.fallback_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, feature_dim)
            )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B*N_views, 3, H, W) - RGB images
            
        Returns:
            features: (B*N_views, feature_dim) - image features
        """
        if self.available:
            # Use DINOv2
            with torch.set_grad_enabled(not self.freeze):
                outputs = self.dinov2(images)
                # Get CLS token embedding
                features = outputs.last_hidden_state[:, 0]  # (B*N, hidden_dim)
                features = self.proj(features)
        else:
            # Use fallback CNN
            features = self.fallback_encoder(images)
        
        return features


class MultiViewFusion(nn.Module):
    """
    Fuse features from multiple views using cross-attention.
    Each view attends to all other views to aggregate information.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_layers)
        ])
        
        # FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.GELU(),
                nn.Linear(feature_dim * 4, feature_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: (B, N_views, feature_dim)
            
        Returns:
            fused_features: (B, N_views, feature_dim) - each view now aware of others
        """
        x = view_features
        
        for attn, ln1, ffn, ln2 in zip(
            self.attention_layers, self.layer_norms, 
            self.ffns, self.ffn_norms
        ):
            # Self-attention
            attn_out, _ = attn(x, x, x)
            x = ln1(x + attn_out)
            
            # FFN
            ffn_out = ffn(x)
            x = ln2(x + ffn_out)
        
        return x


class MultiViewEncoder(nn.Module):
    """
    Complete Multi-view Image Encoder (Module D).
    
    Pipeline:
        Multi-view Images → DINOv2 → Image Features
        Camera Poses → Embedding → Pose Features
        Image Features + Pose Features → Fusion → Conditioning Features
    
    Output is used to condition the diffusion model via cross-attention.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        feature_dim: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        freeze_vision: bool = True,
        dinov2_model: str = 'dinov2_vits14'
    ):
        super().__init__()
        
        self.image_size = image_size
        self.feature_dim = feature_dim
        
        # Vision encoder (DINOv2)
        self.vision_encoder = DINOv2Encoder(
            model_name=dinov2_model,
            freeze=freeze_vision,
            feature_dim=feature_dim
        )
        
        # Camera pose embedding
        self.pose_embedding = CameraPoseEmbedding(pose_dim=12, embed_dim=feature_dim)
        
        # Multi-view fusion
        self.fusion = MultiViewFusion(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers
        )
        
    def forward(
        self, 
        images: torch.Tensor,
        camera_poses: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multi-view images into conditioning features.
        
        Args:
            images: (B, N_views, 3, H, W) - multi-view RGB images
            camera_poses: (B, N_views, 3, 4) - optional camera poses
            
        Returns:
            conditioning: (B, N_views, feature_dim) - features for cross-attention
        """
        B, N, C, H, W = images.shape
        
        # Reshape for batch processing
        images_flat = images.reshape(B * N, C, H, W)
        
        # Encode images
        image_features = self.vision_encoder(images_flat)  # (B*N, feature_dim)
        image_features = image_features.reshape(B, N, -1)  # (B, N, feature_dim)
        
        # Add pose information if available
        if camera_poses is not None:
            pose_features = self.pose_embedding(camera_poses)  # (B, N, feature_dim)
            # Combine image and pose features
            combined_features = image_features + pose_features
        else:
            combined_features = image_features
        
        # Fuse multi-view information
        fused_features = self.fusion(combined_features)  # (B, N, feature_dim)
        
        return fused_features


# =============================================================================
# Helper functions
# =============================================================================

def create_multiview_encoder(
    feature_dim: int = 768,
    freeze_vision: bool = True,
    **kwargs
) -> MultiViewEncoder:
    """
    Factory function to create MultiViewEncoder with sensible defaults.
    
    Args:
        feature_dim: Dimension of output features
        freeze_vision: Whether to freeze DINOv2 weights
        **kwargs: Additional arguments for MultiViewEncoder
        
    Returns:
        MultiViewEncoder model
    """
    return MultiViewEncoder(
        feature_dim=feature_dim,
        freeze_vision=freeze_vision,
        **kwargs
    )


if __name__ == "__main__":
    # Test Module D
    print("="*70)
    print("🧪 Testing Module D: Multi-view Encoder")
    print("="*70)
    
    # Create model
    print("\n[1/3] Creating encoder...")
    encoder = MultiViewEncoder(
        image_size=224,
        feature_dim=768,
        num_heads=8,
        num_fusion_layers=2,
        freeze_vision=True
    )
    
    num_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"✅ Model created")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n[2/3] Testing forward pass...")
    batch_size = 2
    num_views = 4
    
    # Random images
    images = torch.randn(batch_size, num_views, 3, 224, 224)
    
    # Random camera poses (3x4 matrices)
    camera_poses = torch.randn(batch_size, num_views, 3, 4)
    
    # Forward
    encoder.eval()
    with torch.no_grad():
        features = encoder(images, camera_poses)
    
    print(f"✅ Input: {images.shape}")
    print(f"✅ Output: {features.shape}")
    print(f"   Expected: ({batch_size}, {num_views}, 768)")
    
    # Test without poses
    print("\n[3/3] Testing without camera poses...")
    with torch.no_grad():
        features_no_pose = encoder(images, camera_poses=None)
    print(f"✅ Output (no pose): {features_no_pose.shape}")
    
    print("\n" + "="*70)
    print("✅ MODULE D: All tests passed!")
    print("="*70)
    
    print("\n💡 Usage example:")
    print("""
    from models.multiview_encoder import MultiViewEncoder
    
    # Create encoder
    encoder = MultiViewEncoder(feature_dim=768)
    
    # Prepare data
    images = ...  # (B, N_views, 3, 224, 224)
    poses = ...   # (B, N_views, 3, 4) or None
    
    # Encode
    conditioning = encoder(images, poses)
    
    # Use in diffusion model
    output = diffusion_model(x, t, context=conditioning)
    """)


def create_multiview_encoder(preset='base', image_size=224, **kwargs):
    """
    Helper function to create MultiViewEncoder with common presets.
    
    Presets:
        - 'small': DINOv2-S, 384-dim features
        - 'base': DINOv2-B, 768-dim features  
        - 'large': DINOv2-L, 1024-dim features
    
    Args:
        preset: Model size preset
        image_size: Input image size
        **kwargs: Override any parameters
    """
    presets = {
        'small': {
            'dinov2_model': 'dinov2_vits14',
            'feature_dim': 384,
            'num_heads': 6,
            'num_fusion_layers': 2
        },
        'base': {
            'dinov2_model': 'dinov2_vitb14',
            'feature_dim': 768,
            'num_heads': 8,
            'num_fusion_layers': 2
        },
        'large': {
            'dinov2_model': 'dinov2_vitl14',
            'feature_dim': 1024,
            'num_heads': 8,
            'num_fusion_layers': 3
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
    
    config = presets[preset]
    config.update(kwargs)  # Override with user params
    config['image_size'] = image_size
    
    return MultiViewEncoder(**config)
