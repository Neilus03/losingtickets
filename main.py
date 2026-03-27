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
    
    coords, pixels = load_image_dataset(config.IMAGE_PATH, config.IMAGE_SIZE)
    coords = coords.to(device)
    pixels = pixels.to(device)
    
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
    if not os.path.exists(initial_weights_path):
        torch.save(model.state_dict(), initial_weights_path)
        print(f"Saved initial weights to {initial_weights_path}")
    else:
        model.load_state_dict(torch.load(initial_weights_path, map_location=device))
        print(f"Restored existing initial weights from {initial_weights_path}")
    
    # We train the baseline once
    baseline_path = os.path.join("outputs", "baseline", "dense_reconstruction.png")
    if not os.path.exists(baseline_path):
        print("Training Dense Baseline...")
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = torch.nn.MSELoss()
        
        psnr_val = train_model(
            model=model,
            coords=coords,
            pixels=pixels,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config.EPOCHS,
            device=device
        )
        
        img_array = reconstruct_image(model, coords, config.IMAGE_SIZE)
        Image.fromarray(img_array).save(baseline_path)
        print(f"Dense Baseline Training Complete. PSNR: {psnr_val:.2f} dB")
        
        torch.save(model.state_dict(), os.path.join("checkpoints", "dense_trained.pth"))
    else:
        print("Baseline already exists. Skipping baseline training.")
        
    # Phase 5: Run Iterative Pruning
    run_pruning_pipeline(
        "loser", prune_losing_ticket, initial_weights_path, config, coords, pixels,
        lambda: SIREN(hidden_features=config.HIDDEN_FEATURES, hidden_layers=config.HIDDEN_LAYERS, omega_0=config.OMEGA_0)
    )

if __name__ == "__main__":
    main()
