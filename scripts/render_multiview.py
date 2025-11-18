"""
Multi-view Rendering Script
Tự động render multi-view images từ 3D mesh
"""
import numpy as np
import trimesh
import pyrender
import torch
from PIL import Image
from pathlib import Path
import os

# Suppress pyrender warnings
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # or 'osmesa' for headless

class MultiViewRenderer:
    """
    Render 3D mesh từ nhiều góc nhìn
    """
    def __init__(
        self,
        image_size: int = 224,
        num_views: int = 8,
        distance: float = 2.0,
        light_intensity: float = 3.0
    ):
        self.image_size = image_size
        self.num_views = num_views
        self.distance = distance
        self.light_intensity = light_intensity
        
        # Setup scene
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        
        # Setup camera
        self.camera = pyrender.PerspectiveCamera(
            yfov=np.pi / 3.0,
            aspectRatio=1.0
        )
        
        # Setup light
        self.light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=light_intensity
        )
        
        # Renderer
        self.renderer = pyrender.OffscreenRenderer(
            image_size, image_size
        )
    
    def generate_camera_poses(self, num_views: int = None):
        """
        Generate camera poses orbiting around object
        
        Returns:
            poses: (num_views, 4, 4) camera-to-world transforms
        """
        if num_views is None:
            num_views = self.num_views
        
        poses = []
        
        for i in range(num_views):
            # Azimuth angle
            azimuth = 2 * np.pi * i / num_views
            
            # Elevation (thay đổi một chút để diverse hơn)
            elevation = np.pi / 6 + np.sin(azimuth) * 0.1
            
            # Camera position
            x = self.distance * np.cos(azimuth) * np.cos(elevation)
            y = self.distance * np.sin(azimuth) * np.cos(elevation)
            z = self.distance * np.sin(elevation)
            
            camera_pos = np.array([x, y, z])
            
            # Look at origin
            look_at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            
            # Build camera pose matrix
            pose = self._look_at_matrix(camera_pos, look_at, up)
            poses.append(pose)
        
        return np.stack(poses)
    
    def _look_at_matrix(self, eye, center, up):
        """
        Create look-at matrix (camera-to-world)
        """
        f = center - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        mat = np.eye(4)
        mat[0, :3] = s
        mat[1, :3] = u
        mat[2, :3] = -f
        mat[:3, 3] = eye
        
        return mat
    
    def render_multiview(self, mesh_path: str, num_views: int = None):
        """
        Render mesh từ nhiều góc nhìn
        
        Args:
            mesh_path: Path to mesh file
            num_views: Number of views (default: self.num_views)
        
        Returns:
            images: (num_views, 3, H, W) tensor
            poses: (num_views, 3, 4) camera extrinsics
        """
        if num_views is None:
            num_views = self.num_views
        
        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Normalize mesh to unit cube
        mesh = self._normalize_mesh(mesh)
        
        # Convert to pyrender mesh
        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        
        # Generate camera poses
        camera_poses = self.generate_camera_poses(num_views)
        
        images = []
        poses_out = []
        
        for pose in camera_poses:
            # Add mesh to scene
            mesh_node = self.scene.add(mesh_pr)
            
            # Add camera
            cam_node = self.scene.add(self.camera, pose=pose)
            
            # Add light at camera position
            light_pose = pose.copy()
            light_node = self.scene.add(self.light, pose=light_pose)
            
            # Render
            color, depth = self.renderer.render(self.scene)
            
            # Remove nodes
            self.scene.remove_node(mesh_node)
            self.scene.remove_node(cam_node)
            self.scene.remove_node(light_node)
            
            # Convert to tensor
            img = torch.from_numpy(color).float() / 255.0  # (H, W, 3)
            img = img.permute(2, 0, 1)  # (3, H, W)
            images.append(img)
            
            # Extract extrinsic (3x4)
            extrinsic = pose[:3, :]  # (3, 4)
            poses_out.append(extrinsic)
        
        images = torch.stack(images)  # (N, 3, H, W)
        poses_out = torch.from_numpy(np.stack(poses_out)).float()  # (N, 3, 4)
        
        return images, poses_out
    
    def _normalize_mesh(self, mesh):
        """
        Normalize mesh to unit cube centered at origin
        """
        # Center
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        
        # Scale to unit cube
        scale = (mesh.bounds[1] - mesh.bounds[0]).max()
        mesh.vertices /= scale
        
        return mesh
    
    def save_rendered_views(self, mesh_path: str, output_dir: str):
        """
        Render và lưu images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        images, poses = self.render_multiview(mesh_path)
        
        # Save images
        for i, img in enumerate(images):
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(output_dir / f'view_{i:02d}.png')
        
        # Save poses
        np.save(output_dir / 'poses.npy', poses.numpy())
        
        print(f"✅ Saved {len(images)} views to {output_dir}")

def test_rendering():
    """
    Test rendering với một mesh mẫu
    """
    print("Testing multi-view rendering...")
    
    # Create simple mesh for testing
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.export('test_cube.obj')
    
    # Render
    renderer = MultiViewRenderer(
        image_size=224,
        num_views=8
    )
    
    try:
        images, poses = renderer.render_multiview('test_cube.obj')
        
        print(f"✅ Images shape: {images.shape}")
        print(f"✅ Poses shape: {poses.shape}")
        
        # Save visualizations
        renderer.save_rendered_views('test_cube.obj', 'test_renders')
        print("✅ Test passed! Check test_renders/ folder")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install pyrender: pip install pyrender")
        print("2. Install trimesh: pip install trimesh")
        print("3. For headless servers: pip install pyopengl osmesa")
    finally:
        # Cleanup
        if os.path.exists('test_cube.obj'):
            os.remove('test_cube.obj')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, help='Path to mesh file')
    parser.add_argument('--output', type=str, default='renders', help='Output directory')
    parser.add_argument('--num_views', type=int, default=8, help='Number of views')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--test', action='store_true', help='Run test')
    
    args = parser.parse_args()
    
    if args.test:
        test_rendering()
    elif args.mesh:
        renderer = MultiViewRenderer(
            image_size=args.image_size,
            num_views=args.num_views
        )
        renderer.save_rendered_views(args.mesh, args.output)
    else:
        print("Usage:")
        print("  Test: python render_multiview.py --test")
        print("  Render: python render_multiview.py --mesh path/to/mesh.obj --output renders/")
