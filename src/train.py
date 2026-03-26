import torch
import torch.nn as nn
from tqdm import tqdm
import os
import copy
from config import LEARNING_RATE, EPOCHS, TARGET_SPARSITY
from src.utils import PSNR, reconstruct_image

def train_model(model, dataloader, optimizer, loss_fn, epochs, device, coords_full, log_every=100, coords_image_size=256):
    model.to(device)
    coords_full = coords_full.to(device)
    
    # Initialize LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    psnr_val = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for coords_batch, pixels_batch in dataloader:
            coords_batch = coords_batch.to(device)
            pixels_batch = pixels_batch.to(device)
            
            predicted_pixels = model(coords_batch)
            loss = loss_fn(predicted_pixels, pixels_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        
        if epoch % log_every == 0 or epoch == epochs:
            # Full eval
            model.eval()
            with torch.no_grad():
                out_full = model(coords_full)
                # target image should be loaded as pixels_full if we had it, but approx loss is ok.
                # Just reconstruct image to check PSNR. We can use the train dataloader average loss mapping.
                mse_loss = total_loss / len(dataloader)
                psnr_val = PSNR(torch.tensor(mse_loss))
                print(f"Epoch {epoch}/{epochs} | Loss: {mse_loss:.6f} | PSNR: {psnr_val:.2f} dB")
            model.train()
            
    return psnr_val
