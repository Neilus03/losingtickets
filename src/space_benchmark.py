import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from config import HIDDEN_FEATURES, HIDDEN_LAYERS, OMEGA_0, DEVICE
from src.model import SIREN
import torch.nn.utils.prune as prune
import io

def calculate_theoretical_sparse_size(model):
    """
    Calculates the theoretical size using Coordinate (COO) format.
    Each non-zero weight needs:
    - 4 bytes (Float32 value)
    - 4 bytes (Int32 index/position)
    Total = 8 bytes per non-zero.
    """
    total_non_zeros = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # PyTorch pruning stores mask
            mask = module.weight_mask
            non_zeros = (mask != 0).sum().item()
            total_non_zeros += non_zeros
            total_params += mask.numel()
            
    # Theoretical COO Size
    # Non-zeros * 8 bytes (value + index) + Bias (dense)
    # Bias is tiny so we focus on weights
    theoretical_bytes = total_non_zeros * 8
    dense_bytes = total_params * 4
    
    return dense_bytes, theoretical_bytes, total_non_zeros / total_params

def run_space_benchmark():
    print("Mona Lisa Project: Model Compression benchmark")
    print("-" * 60)
    
    folder = "checkpoints"
    results_path = os.path.join("outputs", "space_benchmark.txt")
    
    device = torch.device("cpu")
    
    # 1. Base Model (Dense)
    dense_ckpt = os.path.join(folder, "dense_trained.pth")
    dense_disk_kb = os.path.getsize(dense_ckpt) / 1024
    print(f"DENSE MODEL (100% weights) | Disk Size: {dense_disk_kb:8.2f} KB")
    print("-" * 60)
    
    with open(results_path, "w") as f:
        f.write("Model Storage & Compression Benchmark\n")
        f.write("-" * 60 + "\n")
        f.write(f"Dense Baseline Disk Size: {dense_disk_kb:.2f} KB\n\n")
        
        for t in ["winner", "random", "loser"]:
            ckpts = [chk for chk in os.listdir(folder) if chk.startswith(f"{t}_iter_") and chk.endswith(".pth")]
            ckpts.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
            
            # Select first and last
            if not ckpts: continue
            subset = [ckpts[0], ckpts[len(ckpts)//2], ckpts[-1]]
            
            f.write(f"Ticket Type: {t.upper()}\n")
            for ck in subset:
                # Load
                model = SIREN(hidden_features=HIDDEN_FEATURES, hidden_layers=HIDDEN_LAYERS, omega_0=OMEGA_0).to(device)
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        prune.identity(module, 'weight')
                
                model.load_state_dict(torch.load(os.path.join(folder, ck), map_location=device))
                
                dense_b, sparse_b, ratio = calculate_theoretical_sparse_size(model)
                disk_kb = os.path.getsize(os.path.join(folder, ck)) / 1024
                
                line = (f"Iter {ck.split('_')[2][:2]} | Disk: {disk_kb:7.1f} KB | "
                        f"Weights: {ratio*100:6.2f}% | "
                        f"Theoretical Sparse: {sparse_b/1024:7.1f} KB ({(dense_b/sparse_b):.1f}x compression)\n")
                
                print(line, end="")
                f.write(line)
            f.write("\n")

    print("-" * 60)
    print("EXPLANATION:")
    print("1. Disk Size is currently LARGE for pruned models because PyTorch stores BOTH")
    print("   the original weights AND the mask. This is for training/rewinding ease.")
    print("2. Theoretical Sparse Size shows what happens if you strip the mask and store")
    print("   only non-zeros (using COO format).")
    print(f"\nDetailed report saved to: {results_path}")

if __name__ == "__main__":
    run_space_benchmark()
