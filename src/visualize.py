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
    all_files = [f for f in os.listdir(folder) if f.endswith(".png")]
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
        all_files = [f for f in os.listdir(folder) if f.endswith(".png")]
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
        all_files = [f for f in os.listdir(folder) if f.endswith(".png")]
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
        files = [f for f in os.listdir(folder) if f.endswith(".png")]
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

def main():
    log_file = os.path.join("logs", "experiment_log.csv")
    plot_psnr_curves(log_file)
    create_timeline_strip("winner")
    create_timeline_strip("random")
    create_timeline_strip("loser")
    create_showdown_grid(target_sparsity=5.0)
    generate_gifs()
    generate_interactive_html()

if __name__ == "__main__":
    main()
