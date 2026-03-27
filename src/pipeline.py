import torch
import torch.optim as optim
import copy
import os
import math
from config import LEARNING_RATE, EPOCHS, TARGET_SPARSITY
from src.train import train_model
from src.utils import reconstruct_image, PSNR
from PIL import Image

def run_pruning_pipeline(
    ticket_type,
    prune_function,
    initial_weights_path,
    config,
    coords,
    pixels,
    model_class
):
    print(f"\n=== Starting Pipeline: {ticket_type.upper()} ===")
    
    # Prune Schedule Calculation
    p = config.PRUNE_RATE
    target = config.TARGET_SPARSITY
    import math
    if p >= 1.0 or p <= 0.0:
        raise ValueError("Prune rate must be between 0 and 1 exclusive")
    total_iterations = math.ceil(math.log(target) / math.log(1 - p))
    print(f"Target Sparsity: {target*100:.1f}% -> Requires {total_iterations} iterations")

    device = config.DEVICE
    
    # Step A: Initialize Fresh Model and Load Initial Weights
    model = model_class()
    model.load_state_dict(torch.load(initial_weights_path))
    model.to(device)
    coords = coords.to(device)
    pixels = pixels.to(device)
    image_size = config.IMAGE_SIZE

    # Step B: Iteration Loop
    log_file = os.path.join("logs", "experiment_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("ticket_type,iteration,remaining_pct,psnr_db\n")
            
    # Save Initial Baseline as iteration 0 (if not done globally)
    # We assume iteration 0 is the unpruned baseline.
    
    current_sparsity = 0.0
    start_iter = 1
    
    # Auto-Resume Logic
    last_iter = 0
    for iteration in range(1, total_iterations + 1):
        if os.path.exists(os.path.join("checkpoints", f"{ticket_type}_iter_{iteration}.pth")):
            last_iter = iteration
            
    if last_iter > 0:
        print(f"Discovered checkpoint for {ticket_type} Iteration {last_iter}. Restoring directly to GPU...")
        start_iter = last_iter + 1
        
        # 1. We must structurally prepare the model by adding the PyTorch pruning hooks so the state_dict matches!
        import torch.nn.utils.prune as prune
        import torch.nn as nn
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.custom_from_mask(module, name='weight', mask=torch.ones_like(module.weight))
                
        # 2. Flawlessly load all weights AND the exact pruning mask maps onto the GPU
        model.load_state_dict(torch.load(os.path.join("checkpoints", f"{ticket_type}_iter_{last_iter}.pth"), map_location=device))
        
        from src.pruning import compute_sparsity
        current_sparsity = compute_sparsity(model)
        
        if last_iter == total_iterations:
            print(f"{ticket_type} pipeline entirely completed from disk.")
            return
            
    for iteration in range(start_iter, total_iterations + 1):
        remaining_pct = (1.0 - current_sparsity) * 100
        print(f"\n--- {ticket_type} Iteration {iteration}/{total_iterations} | Remaining: {remaining_pct:.1f}% ---")
        
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = torch.nn.MSELoss()
        
        # B.1 Train
        psnr_val = train_model(
            model=model,
            coords=coords,
            pixels=pixels,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config.EPOCHS,
            device=device
        )
        
        # B.2 Evaluate & Save
        img_array = reconstruct_image(model, coords, image_size)
        img_path = os.path.join("outputs", ticket_type, f"iter_{iteration:02d}_remaining_{remaining_pct:.1f}pct.png")
        Image.fromarray(img_array).save(img_path)
        
        with open(log_file, "a") as f:
            f.write(f"{ticket_type},{iteration},{remaining_pct:.1f},{psnr_val:.4f}\n")
            
        # B.3 Apply Pruning
        model = prune_function(model, config.PRUNE_RATE)
        
        # B.5 Check Loop Continuation
        from src.pruning import compute_sparsity
        current_sparsity = compute_sparsity(model)
        remaining_pct_next = (1.0 - current_sparsity) * 100
        
        # Intermediate checkpoint
        chk_path = os.path.join("checkpoints", f"{ticket_type}_iter_{iteration}.pth")
        torch.save(model.state_dict(), chk_path)
        
        if remaining_pct_next / 100.0 <= target:
            print("Target sparsity reached. Breaking.")
            break
            
        # B.4 Rewind Weights to Initialization
        initial_weights = torch.load(initial_weights_path)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # We need to map module name to state_dict key
                # e.g., net.0.linear -> net.0.linear.weight
                w_key = name + '.weight'
                b_key = name + '.bias'
                
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    # Rewind weight orig
                    try:
                         # The key in initial_weights is 'weight', but in module it's 'weight_orig' 
                         module.weight_orig.data = initial_weights[w_key].to(device) * mask
                    except KeyError:
                         # Maybe the key was already weight_orig if we saved a pruned model?
                         pass
                else:
                    # Should not happen if pruned, but just in case
                    module.weight.data = initial_weights[w_key].to(device)
                    
                if module.bias is not None and b_key in initial_weights:
                    module.bias.data = initial_weights[b_key].to(device)
                    
import os
