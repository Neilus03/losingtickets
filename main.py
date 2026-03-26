import config
import os
import torch
import torch.optim as optim
from PIL import Image

from src.utils import set_seed, reconstruct_image
from src.dataset import load_image_dataset
from src.model import SIREN
from src.train import train_model
from src.pipeline import run_pruning_pipeline
from src.pruning import prune_winning_ticket, prune_random_ticket, prune_losing_ticket

def main():
    set_seed(config.SEED)
    device = config.DEVICE
    
    print("Loading dataset...")
    dataloader, coords_full = load_image_dataset(config.IMAGE_PATH, config.IMAGE_SIZE, config.BATCH_SIZE)
    coords_full = coords_full.to(device)
    
    # Phase 3: Baseline Run & State Capture
    print("Initializing Dense SIREN model...")
    model = SIREN(
        in_features=2, 
        hidden_features=config.HIDDEN_FEATURES, 
        out_features=3, 
        hidden_layers=config.HIDDEN_LAYERS, 
        omega_0=config.OMEGA_0
    ).to(device)
    
    initial_weights_path = os.path.join("checkpoints", "initial_weights.pth")
    torch.save(model.state_dict(), initial_weights_path)
    print(f"Saved initial weights to {initial_weights_path}")
    
    # We train the baseline once
    baseline_path = os.path.join("outputs", "baseline", "dense_reconstruction.png")
    if not os.path.exists(baseline_path):
        print("Training Dense Baseline...")
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = torch.nn.MSELoss()
        
        psnr_val = train_model(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config.EPOCHS,
            device=device,
            coords_full=coords_full,
            log_every=100
        )
        
        img_array = reconstruct_image(model, coords_full, config.IMAGE_SIZE)
        Image.fromarray(img_array).save(baseline_path)
        print(f"Dense Baseline Training Complete. PSNR: {psnr_val:.2f} dB")
        
        torch.save(model.state_dict(), os.path.join("checkpoints", "dense_trained.pth"))
    else:
        print("Baseline already exists. Skipping baseline training.")
        
    # Phase 5: Run Iterative Pruning
    # Winner
    run_pruning_pipeline(
        "winner", prune_winning_ticket, initial_weights_path, config, dataloader, coords_full, 
        lambda: SIREN(hidden_features=config.HIDDEN_FEATURES, hidden_layers=config.HIDDEN_LAYERS, omega_0=config.OMEGA_0)
    )
    
    # Random
    run_pruning_pipeline(
        "random", prune_random_ticket, initial_weights_path, config, dataloader, coords_full,
        lambda: SIREN(hidden_features=config.HIDDEN_FEATURES, hidden_layers=config.HIDDEN_LAYERS, omega_0=config.OMEGA_0)
    )
    
    # Loser
    run_pruning_pipeline(
        "loser", prune_losing_ticket, initial_weights_path, config, dataloader, coords_full,
        lambda: SIREN(hidden_features=config.HIDDEN_FEATURES, hidden_layers=config.HIDDEN_LAYERS, omega_0=config.OMEGA_0)
    )

if __name__ == "__main__":
    main()
