import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import numpy as np
from config import HIDDEN_FEATURES, HIDDEN_LAYERS, OMEGA_0
from src.model import SIREN
import torch.nn.utils.prune as prune

def measure_inference_time(model, device, iterations=20):
    # Benchmark size: 64x64 (4096 points) - Much faster for CPU benchmarking
    x = torch.randn(64*64, 2).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(5): # Warmup
            _ = model(x)
            
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
            
    end_time = time.time()
    return (end_time - start_time) / iterations * 1000

def run_benchmark():
    device = torch.device("cpu") # Benchmarking on CPU for maximum reliability
    print(f"Running Fast Inference Benchmark on {str(device).upper()} (4096 points/forward)...")
    print("-" * 60)
    
    folder = "checkpoints"
    ticket_types = ["winner", "random", "loser"]
    
    # Baseline Dense
    dense_model = SIREN(hidden_features=HIDDEN_FEATURES, hidden_layers=HIDDEN_LAYERS, omega_0=3).to(device)
    dense_time = measure_inference_time(dense_model, device)
    
    results_path = os.path.join("outputs", "benchmark.txt")
    with open(results_path, "w") as f:
        f.write(f"Inference Benchmark Results (CPU, 4096 points)\n")
        f.write("-" * 60 + "\n")
        f.write(f"DENSE BASELINE: {dense_time:8.4f} ms\n\n")
        
        for t in ticket_types:
            ckpts = [f for f in os.listdir(folder) if f.startswith(f"{t}_iter_") and f.endswith(".pth")]
            ckpts.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
            
            indices = [0, len(ckpts)//2, len(ckpts)-1]
            subset = [ckpts[i] for i in indices if i < len(ckpts)]
            
            f.write(f"Stage: {t.upper()}\n")
            for ck in subset:
                model = SIREN(hidden_features=HIDDEN_FEATURES, hidden_layers=HIDDEN_LAYERS, omega_0=OMEGA_0).to(device)
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        prune.identity(module, 'weight')
                
                model.load_state_dict(torch.load(os.path.join(folder, ck), map_location=device))
                
                non_zeros = 0
                total_params = 0
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Count non-zeros in the mask (the actual pruning decision)
                        non_zeros += (module.weight_mask != 0).sum().item()
                        total_params += module.weight_mask.numel()
                
                sparsity = (non_zeros / total_params) * 100
                inf_time = measure_inference_time(model, device)
                
                line = f"[{t.upper():7s}] Iter {ck.split('_')[2][:2]} | Weights Remaining: {sparsity:6.2f}% | Latency: {inf_time:8.4f} ms\n"
                f.write(line)
                print(line, end="")
            f.write("\n")

    print(f"\nBenchmark Complete. Results saved to {results_path}")

    print("-" * 60)
    print("ANALYSIS: As expected, PyTorch unstructured pruning does NOT naturally speed up inference.")
    print("The mask multiplication ($W_{orig} \odot Mask$) remains a dense operation.")
    print("Real speedups require Hardware-Aware Sparse Kernels or Structural Pruning.")

if __name__ == "__main__":
    run_benchmark()
