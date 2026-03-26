import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

def load_image_dataset(image_path, image_size, batch_size):
    # 1. Load and Preprocess Image
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img) # Shape: (3, H, W) in [0.0, 1.0]
    
    # Normalize to [-1.0, 1.0]
    img_tensor = (img_tensor * 2.0) - 1.0 
    
    # 2. Generate Spatial Grid
    x_coords = torch.linspace(-1, 1, image_size)
    y_coords = torch.linspace(-1, 1, image_size)
    
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coordinates = torch.stack([grid_x, grid_y], dim=-1) # Shape: (H, W, 2)
    coordinates = coordinates.reshape(-1, 2)            # Shape: (H*W, 2)
    
    # 3. Flatten the Image
    img_flat = img_tensor.permute(1, 2, 0)              # Shape: (H, W, C)
    img_flat = img_flat.reshape(-1, 3)                  # Shape: (H*W, 3)

    # 4. Create DataLoader
    dataset = TensorDataset(coordinates, img_flat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    
    return dataloader, coordinates
