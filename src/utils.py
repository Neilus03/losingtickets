import torch
import numpy as np
import random

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def PSNR(mse):
    # If targets are [-1, 1], maximum difference is 2.0 -> 2.0^2 = 4.0
    if mse == 0:
        return float('inf')
    return 10 * np.log10(4.0 / mse.item())

def reconstruct_image(model, coords_full, image_size):
    model.eval()
    with torch.no_grad():
        output = model(coords_full)
        output = output.reshape(image_size, image_size, 3)
        # Un-normalize from [-1, 1] to [0, 1]
        output = (output + 1.0) / 2.0
        output = torch.clamp(output, 0, 1)
        output = (output * 255).to(torch.uint8)
    model.train()
    return output.cpu().numpy()
