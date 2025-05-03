"""
3D Model Generator (Text & Image Input) with Visualization
"""
import argparse
import torch
import trimesh
import pyrender
import numpy as np
from io import BytesIO
from PIL import Image
from rembg import remove
from torchvision.transforms import Compose, Resize, ToTensor

# Shap-E components
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

class ShapEGenerator:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.xm = None
        self.model = None
        self.diffusion = None
        
    def initialize_models(self, model_type='text'):
        """Load appropriate models based on input type"""
        self.xm = load_model('transmitter', device=self.device).to(self.device)
        model_name = 'text300M' if model_type == 'text' else 'image300M'
        self.model = load_model(model_name, device=self.device).to(self.device)
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        print(f"Initialized {model_type} model on {self.device}")

    @staticmethod
    def remove_background(input_image: Image.Image) -> Image.Image:
        """Remove image background with proper byte handling"""
        with BytesIO() as buffer:
            input_image.save(buffer, format='PNG')
            output_bytes = remove(buffer.getvalue())
        return Image.open(BytesIO(output_bytes)).convert('RGBA')

    def process_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for Shap-E model"""
    # Load image and remove background
        input_image = Image.open(image_path)
        processed_image = self.remove_background(input_image)  # Apply background removal

    # Convert to RGB and resize to square
        image = processed_image.convert('RGB').resize((256, 256))
    
    # Convert to properly formatted tensor
        transform = Compose([
            ToTensor(),  # Normalizes to [0,1] and converts to (C, H, W)
        ])
    
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(torch.float32).to(self.device)

    def generate_latents(self, input_data, model_type='text', batch_size=4):
        if model_type == 'image' and input_data.shape[1] != 3:
            raise ValueError(f"Image tensor must have 3 channels (RGB). Got: {input_data.shape}")
        """Generate latent vectors from input"""
        common_params = {
            'batch_size': batch_size,
            'model': self.model,
            'diffusion': self.diffusion,
            'guidance_scale': 15.0 if model_type == 'text' else 3.0,
            'progress': True,
            'clip_denoised': True,
            'use_fp16': True,
            'use_karras': True,
            'karras_steps': 64,
            'sigma_min': 1e-3,
            'sigma_max': 160,
            's_churn': 0,
        }

        if model_type == 'text':
            return sample_latents(
                **common_params,
                model_kwargs=dict(texts=[input_data] * batch_size)
            )
        return sample_latents(
            **common_params,
            model_kwargs=dict(images=[input_data] * batch_size)
        )

    def generate_mesh(self, latent):
        """Convert latent to 3D mesh"""
        tri_mesh = decode_latent_mesh(self.xm, latent)
        mesh = tri_mesh.tri_mesh() if hasattr(tri_mesh, 'tri_mesh') else tri_mesh
        return trimesh.Trimesh(
            vertices=np.array(mesh.verts),
            faces=np.array(mesh.faces),
            vertex_colors=mesh.vertex_channels.get('RGB', None)
        )

    def postprocess_mesh(self, mesh):
        """Clean and optimize generated mesh"""
        processed = mesh.copy()
    
        # Replace deprecated methods
        processed.update_faces(processed.unique_faces())  
        processed.remove_unreferenced_vertices()
        processed.update_faces(processed.nondegenerate_faces())  
    
        trimesh.smoothing.filter_laplacian(processed, iterations=2)
        processed.fix_normals()
        return processed

    def visualize_mesh(self, mesh_path: str):
        """
        Visualize the generated mesh using pyrender (interactive window).
        Falls back to trimesh viewer if pyrender fails.
        """
        try:
            mesh = trimesh.load(mesh_path)

            scene = pyrender.Scene()
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(pyrender_mesh)

            # Add camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            cam_pose = np.array([
                [1.0, 0.0,  0.0,  0.0],
                [0.0, 1.0,  0.0, -0.1],
                [0.0, 0.0,  1.0,  0.3],
                [0.0, 0.0,  0.0,  1.0]
            ])
            scene.add(camera, pose=cam_pose)

            # Add light
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            scene.add(light, pose=np.eye(4))

            pyrender.Viewer(scene, use_raymond_lighting=True)

        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Falling back to trimesh viewer.")
            mesh = trimesh.load(mesh_path)
            mesh.show()


def main():
    parser = argparse.ArgumentParser(description='3D Model Generator')
    parser.add_argument('--input', required=True, help='Image path or text prompt')
    parser.add_argument('--output', default='output.obj', help='Output file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of variants')
    parser.add_argument('--visualize', action='store_true', help='Enable auto-visualization')
    args = parser.parse_args()

    generator = ShapEGenerator()
    
    try:
        if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
            generator.initialize_models(model_type='image')
            img_tensor = generator.process_image(args.input)
            latents = generator.generate_latents(img_tensor, model_type='image', batch_size=args.batch_size)
        else:
            generator.initialize_models(model_type='text')
            latents = generator.generate_latents(args.input, model_type='text', batch_size=args.batch_size)

        for i, latent in enumerate(latents):
            mesh = generator.generate_mesh(latent)
            processed_mesh = generator.postprocess_mesh(mesh)
            output_path = args.output.replace('.', f'_{i}.')
            processed_mesh.export(output_path)
            print(f"Saved 3D model to {output_path}")
            
            if args.visualize:
                generator.visualize_mesh(output_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()