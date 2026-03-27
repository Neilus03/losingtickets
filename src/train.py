import torch
import torch.nn as nn
from tqdm import tqdm
import os
import copy
from config import LEARNING_RATE, EPOCHS, TARGET_SPARSITY
from src.utils import PSNR, reconstruct_image

def train_model(model, coords, pixels, optimizer, loss_fn, epochs, device, log_every=5):
    model.train()
    model.to(device)
    x = coords.to(device)
    y = pixels.to(device)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | PSNR: {PSNR(loss):.2f} dB", flush=True)
                
    return PSNR(loss)
