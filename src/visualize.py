import os
import sys

# Ensure root directory is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import json
from PIL import Image, ImageDraw, ImageFont
from config import IMAGE_SIZE

def plot_psnr_curves(log_file):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    df = pd.read_csv(log_file)

    plt.style.use('dark_background')
    plt.figure(figsize=(12, 7))

    colors = {'winner': '#00ffcc', 'random': '#ffcc00', 'loser': '#ff3366'}
    for name, group in df.groupby("ticket_type"):
        plt.plot(group["remaining_pct"], group["psnr_db"],
                 marker='o', label=name.capitalize(),
                 linewidth=3, markersize=8, color=colors.get(name, '#ffffff'))

    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.title("The Mona Lisa Lottery: PSNR vs Sparsity", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Remaining Weights (%)", fontsize=14)
    plt.ylabel("PSNR (dB)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, loc='lower left')

    out_path = os.path.join("outputs", "plots", "psnr_vs_sparsity.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"Saved highly-styled PSNR plot to {out_path}")

def extract_sparsity_from_filename(filename):
    # E.g. iter_13_remaining_5.0pct.png -> 5.0
    try:
        parts = filename.split('_remaining_')
        pct_str = parts[1].replace('pct.png', '')
        return float(pct_str)
    except:
        return 100.0

def create_timeline_strip(ticket_type, sparsities_to_plot=[100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0]):
    folder = os.path.join("outputs", ticket_type)
    if not os.path.exists(folder): return

    # Find images best matching the target sparsities
    all_files = [f for f in os.listdir(folder) if f.startswith("iter_") and f.endswith(".png")]
    if not all_files: return

    available_sparsities = {extract_sparsity_from_filename(f): f for f in all_files}

    selected_files = []
    for target in sparsities_to_plot:
        # get closest
        closest_sparsity = min(available_sparsities.keys(), key=lambda k: abs(k - target))
        closest_file = available_sparsities[closest_sparsity]
        # Ensure we don't add the same file twice if multiple targets map to it
        if closest_file not in selected_files:
            selected_files.append(closest_file)

    if not selected_files: return
    # Stitch horizontally
    widths, heights = zip(*(Image.open(os.path.join(folder, f)).size for f in selected_files))

    total_width = sum(widths) + (10 * (len(selected_files) - 1))
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height + 60), color='#121212')

    draw = ImageDraw.Draw(new_im)

    x_offset = 0
    for file, w in zip(selected_files, widths):
        im = Image.open(os.path.join(folder, file))
        new_im.paste(im, (x_offset, 60))
        pct = extract_sparsity_from_filename(file)
        fill_color = "#00ffcc" if ticket_type=="winner" else "#ffcc00" if ticket_type=="random" else "#ff3366"
        draw.text((x_offset + 10, 20), f"{pct:.1f}%", fill=fill_color)
        x_offset += w + 10

    out_path = os.path.join("outputs", "plots", f"timeline_{ticket_type}.png")
    new_im.save(out_path)
    print(f"Saved timeline to {out_path}")

def create_showdown_grid(target_sparsity=5.0):
    tickets = ["winner", "random", "loser"]
    imgs = []

    # Target image as reference
    target_path = os.path.join("data", "target_image.jpg")
    if os.path.exists(target_path):
        im = Image.open(target_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        imgs.append(("Original (100%)", im))

    for t in tickets:
        folder = os.path.join("outputs", t)
        if not os.path.exists(folder): continue
        all_files = [f for f in os.listdir(folder) if f.startswith("iter_") and f.endswith(".png")]
        if not all_files: continue
        available_sparsities = {extract_sparsity_from_filename(f): f for f in all_files}
        closest = min(available_sparsities.keys(), key=lambda k: abs(k - target_sparsity))
        im = Image.open(os.path.join(folder, available_sparsities[closest]))
        imgs.append((f"{t.capitalize()} (~{closest:.1f}%)", im))

    if not imgs: return

    widths, heights = zip(*(i[1].size for i in imgs))
    total_width = sum(widths) + 20 * (len(imgs) - 1)

    new_im = Image.new('RGB', (total_width, max(heights) + 50), color='#0a0a0a')
    draw = ImageDraw.Draw(new_im)

    x_offset = 0
    for label, im in imgs:
        new_im.paste(im, (x_offset, 50))
        draw.text((x_offset + 10, 15), label, fill="white")
        x_offset += im.width + 20

    out_path = os.path.join("outputs", "plots", "showdown_grid.png")
    new_im.save(out_path)
    print(f"Saved Showdown Grid to {out_path}")

def generate_gifs():

    for ticket_type in ["winner", "random", "loser"]:
        folder = os.path.join("outputs", ticket_type)
        if not os.path.exists(folder): continue
        all_files = [f for f in os.listdir(folder) if f.startswith("iter_") and f.endswith(".png")]
        if not all_files: continue

        # Sort by iteration (filename has iter_XX)
        all_files.sort(key=lambda x: int(x.split('_')[1]))

        images = []
        for f in all_files:
            pil_img = Image.open(os.path.join(folder, f)).convert('RGB')
            draw = ImageDraw.Draw(pil_img)
            pct = extract_sparsity_from_filename(f)
            draw.rectangle(((5, 5), (120, 30)), fill="black")
            draw.text((10, 10), f"Sparsity: {pct:.2f}%", fill="#ffffff")
            images.append(np.array(pil_img))

        out_path = os.path.join("outputs", "plots", f"degradation_{ticket_type}.gif")
        if images:
            imageio.mimsave(out_path, images, duration=0.8, loop=0)
            print(f"Saved enhanced GIF to {out_path}")

def generate_interactive_html():

    data_map = {}
    for t in ["winner", "random", "loser"]:
        folder = os.path.join("outputs", t)
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if f.startswith("iter_") and f.endswith(".png")]
        files.sort(key=lambda x: int(x.split('_')[1]))
        data_map[t] = [f"../{t}/{f}" for f in files]

    # Assume iteration count is the same for all (or bounded by winner)
    iters = len(data_map.get("winner", []))
    if iters == 0: return

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mona Lisa Lottery Interactive Viewer</title>
    <style>
        body {{ background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; text-align: center; margin: 0; padding: 2rem; }}
        h1 {{ background: linear-gradient(90deg, #00ffcc, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .grid {{ display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; }}
        .card {{ background: #1e293b; padding: 1rem; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }}
        img {{ border-radius: 8px; max-width: 100%; border: 2px solid #334155; transition: transform 0.2s; }}
        img:hover {{ transform: scale(1.02); }}
        input[type=range] {{ width: 60%; margin: 2rem 0; accent-color: #00ffcc; }}
        h3 {{ margin-top: 0; }}
        .iteration-label {{ font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: #ffcc00; }}
    </style>
</head>
<body>
    <h1>The Mona Lisa Lottery Visualization</h1>
    <p>Slide to prune the Implicit Neural Representation!</p>
    <div class="iteration-label" id="stepLabel">Iteration: 1</div>
    <input type="range" id="slider" min="0" max="{iters-1}" value="0" oninput="updateImages()">

    <div class="grid">
        <div class="card">
            <h3>Winning Ticket</h3>
            <img id="winner" src="{data_map.get('winner', [''])[0]}" alt="Winner">
        </div>
        <div class="card">
            <h3>Random Ticket</h3>
            <img id="random" src="{data_map.get('random', [''])[0]}" alt="Random">
        </div>
        <div class="card">
            <h3>Losing Ticket</h3>
            <img id="loser" src="{data_map.get('loser', [''])[0]}" alt="Loser">
        </div>
    </div>
    <script>
        const data = {json.dumps(data_map)};
        function updateImages() {{
            const val = document.getElementById('slider').value;
            document.getElementById('stepLabel').innerText = "Algorithm Iteration: " + (parseInt(val) + 1);
            if(data['winner'] && data['winner'][val]) document.getElementById('winner').src = data['winner'][val];
            if(data['random'] && data['random'][val]) document.getElementById('random').src = data['random'][val];
            if(data['loser'] && data['loser'][val]) document.getElementById('loser').src = data['loser'][val];
        }}
        // Init
        updateImages();
    </script>
</body>
</html>
"""
    out_path = os.path.join("outputs", "plots", "index.html")
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"Saved interactive HTML to {out_path}")

def plot_spectral_comparison():
    """
    Empirically validates the frequency destruction hypothesis by comparing 2D FFTs
    across ALL available iterations (granularities).
    """
    tickets = ["winner", "random", "loser"]
    
    # Original image for reference
    target_path = os.path.join("data", "target_image.jpg")
    if not os.path.exists(target_path):
        print("Original image not found for spectral analysis.")
        return
    
    orig_pil = Image.open(target_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    orig_np = np.array(orig_pil.convert("L")) # Grayscale for FFT
    
    f = np.fft.fft2(orig_np)
    fshift = np.fft.fftshift(f)
    orig_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Identify all available iterations
    # We use the winner folder as the reference for iterations
    winner_folder = os.path.join("outputs", "winner")
    if not os.path.exists(winner_folder): return
    # Filter to only get the base reconstructions, not the fft overlays
    all_files = sorted([f for f in os.listdir(winner_folder) if f.startswith("iter_") and f.endswith(".png")],
                      key=lambda x: int(x.split('_')[1]))
    
    gif_frames = []
    
    for filename in all_files:
        iter_num = int(filename.split('_')[1])
        pct = extract_sparsity_from_filename(filename)
        
        imgs_spatial = [orig_pil]
        imgs_freq = [orig_spectrum]
        labels = ["Original (100%)"]
        
        for t in tickets:
            folder = os.path.join("outputs", t)
            # Find the file with the same iteration number
            match = [f for f in os.listdir(folder) if f.startswith(f"iter_{iter_num:02d}") and f.endswith(".png")]
            if not match: continue
            
            recon_file = match[0]
            recon_pct = extract_sparsity_from_filename(recon_file)
            
            recon_pil = Image.open(os.path.join(folder, recon_file)).convert("RGB")
            recon_np = np.array(recon_pil.convert("L"))
            
            # Compute FFT
            f_recon = np.fft.fft2(recon_np)
            fshift_recon = np.fft.fftshift(f_recon)
            spectrum_recon = 20 * np.log(np.abs(fshift_recon) + 1)
            
            # Save individual FFT image for the web lab overlay
            # Normalize to 0-255 for saving
            spec_min, spec_max = spectrum_recon.min(), spectrum_recon.max()
            spec_norm = (spectrum_recon - spec_min) / (spec_max - spec_min + 1e-8)
            spec_img = Image.fromarray((spec_norm * 255).astype(np.uint8)).convert("L")
            
            # Save to both outputs/ and docs/ so the web lab works
            for base_dir in ["outputs", "docs"]:
                out_folder = os.path.join(base_dir, t)
                if os.path.exists(out_folder):
                    fft_out_path = os.path.join(out_folder, f"fft_iter_{iter_num:02d}.png")
                    spec_img.save(fft_out_path)
            
            imgs_spatial.append(recon_pil)
            imgs_freq.append(spectrum_recon)
            labels.append(f"{t.capitalize()} ({recon_pct:.1f}%)")

        # Plot 2x4 grid
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        
        for i in range(4):
            # Spatial row
            axes[0, i].imshow(imgs_spatial[i])
            axes[0, i].set_title(labels[i], fontsize=12, pad=10)
            axes[0, i].axis('off')
            
            # Frequency row
            axes[1, i].imshow(imgs_freq[i], cmap='magma')
            axes[1, i].set_title("Magnitude Spectrum", fontsize=10, pad=5)
            axes[1, i].axis('off')

        fig.suptitle(f"The Mona Lisa Lottery: Spectral Collapse (Iteration {iter_num})", 
                     fontsize=20, fontweight='bold', color='#00ffcc', y=0.96)
        
        # Save frame buffer for GIF if needed
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(frame)
        
        # Save individual file for last few steps or key granularities? 
        # Actually, let's just save them all as requested "at all granularities"
        out_path = os.path.join("outputs", "plots", f"spectral_iter_{iter_num:02d}.png")
        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        print(f"Saved spectral comparison for iteration {iter_num} to {out_path}")

    # Generate GIF
    if gif_frames:
        gif_path = os.path.join("outputs", "plots", "spectral_collapse_evolution.gif")
        imageio.mimsave(gif_path, gif_frames, duration=0.8, loop=0)
        print(f"Saved spectral evolution GIF to {gif_path}")

def calculate_high_frequency_energy(image_np, radius_ratio=0.1):
    """
    Calculates the energy in high-frequency components of the 2D FFT.
    """
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_ratio)
    
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    
    # HFE is the sum of magnitudes OUTSIDE the central mask
    hfe = np.sum(mag[~mask])
    return hfe

def plot_spectral_energy_curves():
    """
    Plots the High-Frequency Energy (HFE) retention across pruning iterations.
    """
    tickets = ["winner", "random", "loser"]
    colors = {'winner': '#00ffcc', 'random': '#ffcc00', 'loser': '#ff3366'}
    
    target_path = os.path.join("data", "target_image.jpg")
    orig_pil = Image.open(target_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    orig_np = np.array(orig_pil)
    orig_hfe = calculate_high_frequency_energy(orig_np)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    all_hfe_data = []

    for t in tickets:
        folder = os.path.join("outputs", t)
        if not os.path.exists(folder): continue
        # Filter to only get the base reconstructions, not the fft overlays
        all_files = sorted([f for f in os.listdir(folder) if f.startswith("iter_") and f.endswith(".png")],
                          key=lambda x: int(x.split('_')[1]))
        
        remaining_pcts = []
        hfe_retentions = []
        
        for f in all_files:
            pct = extract_sparsity_from_filename(f)
            recon_pil = Image.open(os.path.join(folder, f)).convert("L")
            recon_np = np.array(recon_pil)
            recon_hfe = calculate_high_frequency_energy(recon_np)
            
            retention = (recon_hfe / orig_hfe) * 100
            remaining_pcts.append(pct)
            hfe_retentions.append(retention)
            
            all_hfe_data.append({
                "ticket_type": t,
                "remaining_pct": pct,
                "hfe_retention_pct": retention
            })
            
        plt.plot(remaining_pcts, hfe_retentions, marker='o', label=t.capitalize(),
                 linewidth=2.5, markersize=5, color=colors.get(t, '#ffffff'))
                 
    # Save quantitative data to CSV
    csv_path = os.path.join("logs", "spectral_analysis.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(all_hfe_data).to_csv(csv_path, index=False)
    print(f"Saved quantitative spectral data to {csv_path}")

    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.title("Frequency Domain Collapse: High-Frequency Energy Retention", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Remaining Weights (%)", fontsize=12)
    plt.ylabel("High-Frequency Energy Retention (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=10)
    
    out_path = os.path.join("outputs", "plots", "spectral_energy_vs_sparsity.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"Saved spectral energy chart to {out_path}")

def main():
    log_file = os.path.join("logs", "experiment_log.csv")
    plot_psnr_curves(log_file)
    create_timeline_strip("winner")
    create_timeline_strip("random")
    create_timeline_strip("loser")
    create_showdown_grid(target_sparsity=5.0)
    generate_gifs()
    generate_interactive_html()
    plot_spectral_comparison()
    plot_spectral_energy_curves()

if __name__ == "__main__":
    main()
