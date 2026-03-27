import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

def load_image_dataset(image_path, image_size, batch_size=None):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    img_tensor = (transform(img) * 2.0) - 1.0 
    
    x_coords = torch.linspace(-1, 1, image_size)
    y_coords = torch.linspace(-1, 1, image_size)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coordinates = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    
    img_flat = img_tensor.permute(1, 2, 0).reshape(-1, 3)
    
    return coordinates, img_flat
