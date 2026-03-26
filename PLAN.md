# Losing Tickets in Neural Representations

## Complete Implementation Plan

> **Project Goal:** Train a SIREN (Sinusoidal Representation Network) to memorize a single image as an Implicit Neural Representation (INR), then systematically apply Iterative Magnitude Pruning (IMP) to discover "Winning Tickets," "Random Tickets," and — most importantly — "Losing Tickets." The final deliverable is a set of stunning visual artifacts that show how a neural network's internal structure encodes (and fails to encode) a single image.

---

## Table of Contents

1. [Phase 0 — Environment & Project Scaffolding](#phase-0--environment--project-scaffolding)
2. [Phase 1 — Data Pipeline (Coordinate Generation)](#phase-1--data-pipeline-coordinate-generation)
3. [Phase 2 — Architecture Construction (The SIREN)](#phase-2--architecture-construction-the-siren)
4. [Phase 3 — The Baseline Run & State Capture](#phase-3--the-baseline-run--state-capture)
5. [Phase 4 — The Pruning Engine (The Logic Core)](#phase-4--the-pruning-engine-the-logic-core)
6. [Phase 5 — The Iterative Pruning Pipeline (The Experiment Loop)](#phase-5--the-iterative-pruning-pipeline-the-experiment-loop)
7. [Phase 6 — Extraction, Visualization & Assembly](#phase-6--extraction-visualization--assembly)

---

## Phase 0 — Environment & Project Scaffolding

Before you write a single line of model code, you must set up a reproducible, well-organized project environment. Skipping this step leads to chaos later when you have hundreds of saved images and checkpoints scattered everywhere.

### 0.1 — Python Environment

1. **Create a dedicated virtual environment.** Use `conda` or `venv`. Name it something descriptive (e.g., `losingtickets`).

   ```
   conda create -n losingtickets python=3.10 -y
   conda activate losingtickets
   ```

   or, with `venv`:

   ```
   python -m venv .venv
   source .venv/Scripts/activate   # Windows
   ```

2. **Install the required packages.** You need _exactly_ these libraries — nothing more, nothing less for the core experiment:

   | Package | Purpose | Minimum Version |
   |---------|---------|-----------------|
   | `torch` | Core deep learning framework | ≥ 2.0 |
   | `torchvision` | Image loading and transforms | ≥ 0.15 |
   | `Pillow` (PIL) | Low-level image I/O | ≥ 9.0 |
   | `numpy` | Numerical operations | ≥ 1.24 |
   | `matplotlib` | Plotting PSNR curves and image grids | ≥ 3.7 |
   | `tqdm` | Progress bars for training loops | ≥ 4.65 |
   | `imageio` | GIF generation for degradation animations | ≥ 2.31 |

   Install them all in one command:

   ```
   pip install torch torchvision Pillow numpy matplotlib tqdm imageio
   ```

3. **Verify CUDA availability.** Run a quick check to confirm GPU access:

   ```python
   import torch
   print(torch.cuda.is_available())   # Must print True
   print(torch.cuda.get_device_name(0))
   ```

   If this prints `False`, troubleshoot your CUDA/cuDNN installation before going further. The experiment _can_ run on CPU, but it will be 10–50× slower.

4. **Set all random seeds immediately.** Reproducibility is non-negotiable in this experiment because the Lottery Ticket Hypothesis depends on comparing runs that start from identical initialization. At the very top of your main script (or in a `utils.py`), add:

   ```python
   import torch
   import numpy as np
   import random

   SEED = 42

   def set_seed(seed=SEED):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```

   **Call `set_seed()` at the very start of every script** before any tensors are created.

### 0.2 — Project Directory Structure

Create the following directory tree _before you start coding_. Every single output of this experiment has a place to live:

```
losingtickets/
├── PLAN.md                  # This file
├── README.md                # Project description
├── requirements.txt         # Pin all dependency versions
├── config.py                # ALL hyperparameters in one place
├── data/
│   └── target_image.jpg     # The single image you will memorize
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Phase 1: Coordinate dataset
│   ├── model.py             # Phase 2: SIREN architecture
│   ├── train.py             # Phase 3: Training loop
│   ├── pruning.py           # Phase 4: Pruning functions
│   ├── pipeline.py          # Phase 5: IMP pipeline orchestrator
│   ├── visualize.py         # Phase 6: Plotting and image grids
│   └── utils.py             # Seed, PSNR, image save helpers
├── checkpoints/
│   └── initial_weights.pth  # The sacred initial state_dict
├── outputs/
│   ├── baseline/            # Dense model reconstruction
│   ├── winner/              # Winning ticket images at each sparsity
│   ├── random/              # Random ticket images at each sparsity
│   ├── loser/               # Losing ticket images at each sparsity
│   └── plots/               # PSNR curves, comparison grids, GIFs
└── logs/
    └── experiment_log.csv   # Structured log: sparsity, PSNR, ticket_type
```

### 0.3 — The Configuration File (`config.py`)

Centralize **every single hyperparameter** in one file. Never hard-code values inside training loops. This makes it trivial to re-run the experiment with different settings.

The file should define (at minimum) these variables:

| Variable Name | Default Value | Description |
|---|---|---|
| `IMAGE_PATH` | `"data/target_image.jpg"` | Path to the target image |
| `IMAGE_SIZE` | `256` | Resize target to this square resolution |
| `HIDDEN_FEATURES` | `256` | Number of neurons per hidden layer |
| `HIDDEN_LAYERS` | `4` | Number of hidden layers in the SIREN |
| `OMEGA_0` | `30.0` | Frequency scaling for SIREN sine activations |
| `LEARNING_RATE` | `1e-4` | Adam optimizer learning rate |
| `EPOCHS` | `1000` | Number of training epochs per run |
| `BATCH_SIZE` | `8192` | Pixels per training batch |
| `PRUNE_RATE` | `0.20` | Fraction of _remaining_ weights to prune per iteration |
| `TARGET_SPARSITY` | `0.02` | Stop pruning when this fraction of weights remains |
| `SEED` | `42` | Global random seed |
| `DEVICE` | `"cuda"` | Device string (auto-fallback to `"cpu"`) |

### 0.4 — Choose Your Target Image

Pick **one** image. Some suggestions:

- **The Mona Lisa** — iconic, immediately recognizable, rich in fine detail.
- **A personal photo** — your dog, your face, a landscape. Makes the results more personal.
- **A high-contrast pattern** — like a chess board or fractal. Useful for understanding frequency decomposition.

**Requirements for the image:**

- It must be square, or you must crop it to a square before placing it in `data/`.
- Resolution should be at least 256×256 pixels. You will resize it programmatically, but starting below this means you lose detail.
- It should have a mix of **low-frequency regions** (smooth gradients, sky, skin) and **high-frequency regions** (edges, text, hair, patterns). This mix is what makes the pruning results visually interesting.

---

## Phase 1 — Data Pipeline (Coordinate Generation)

Unlike standard image classification where your dataset is a folder of labeled images, in an Implicit Neural Representation the "dataset" is a mathematical mapping from **pixel coordinates** `(x, y)` to **color values** `(r, g, b)`. You are teaching the network a continuous function: `f(x, y) → (r, g, b)`.

### 1.1 — Load and Preprocess the Target Image

**File:** `src/dataset.py`

1. **Load the image** from disk using `PIL.Image.open()`. Convert it to RGB mode explicitly with `.convert('RGB')` — this handles grayscale images or images with alpha channels gracefully.

2. **Resize to a square.** Use `torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))` where `IMAGE_SIZE` comes from your config. Use `InterpolationMode.BILINEAR` for the resize to avoid aliasing artifacts.

3. **Convert to a PyTorch tensor.** Use `torchvision.transforms.ToTensor()`. This automatically converts pixel values from `[0, 255]` integers to `[0.0, 1.0]` floats and rearranges dimensions from `(H, W, C)` to `(C, H, W)`.

4. **Normalize to `[-1.0, 1.0]` range.** This is **critical** for SIREN networks. The `tanh` output layer (or the unbounded linear output) needs targets centered around zero to converge smoothly. Apply the transform:

   ```
   pixel_value_normalized = (pixel_value * 2.0) - 1.0
   ```

   So `0.0` becomes `-1.0`, `0.5` becomes `0.0`, and `1.0` becomes `1.0`.

5. **Resulting tensor shape**: `(3, IMAGE_SIZE, IMAGE_SIZE)` with values in `[-1.0, 1.0]`.

### 1.2 — Generate the Spatial Coordinate Grid (The Inputs)

This is the "X" in your supervised learning setup. Each input is a 2D coordinate telling the network _which pixel to paint_.

1. **Create 1D coordinate vectors.** Generate two 1D tensors, each of length `IMAGE_SIZE`, linearly spaced from `-1.0` to `1.0`:

   ```
   x_coords = torch.linspace(-1, 1, IMAGE_SIZE)
   y_coords = torch.linspace(-1, 1, IMAGE_SIZE)
   ```

2. **Create the 2D meshgrid.** Use `torch.meshgrid(y_coords, x_coords, indexing='ij')`. This returns two tensors, each of shape `(IMAGE_SIZE, IMAGE_SIZE)`:
   - `grid_y`: each row is the same y-coordinate.
   - `grid_x`: each column is the same x-coordinate.

   > **Why `indexing='ij'`?** This matches the image convention where the first dimension is the row (y-axis, top to bottom) and the second dimension is the column (x-axis, left to right). Using `'xy'` indexing would swap your image axes and result in a transposed reconstruction.

3. **Stack and flatten.** Stack the two grids along the last dimension to get a `(IMAGE_SIZE, IMAGE_SIZE, 2)` tensor, then reshape to `(IMAGE_SIZE * IMAGE_SIZE, 2)`:

   ```
   coordinates = torch.stack([grid_x, grid_y], dim=-1)  # Shape: (H, W, 2)
   coordinates = coordinates.reshape(-1, 2)               # Shape: (H*W, 2)
   ```

   For a 256×256 image, this is **65,536 coordinate pairs**. Each row is one `(x, y)` pair.

### 1.3 — Flatten the Target Image (The Labels)

This is the "Y" in your supervised learning setup. Each label is the RGB color at the corresponding coordinate.

1. **Rearrange the image tensor** from `(C, H, W)` to `(H, W, C)` using `.permute(1, 2, 0)`.

2. **Flatten** to `(H * W, 3)` using `.reshape(-1, 3)`.

3. **Verify alignment.** The pixel at row `i`, column `j` in the image must correspond to the coordinate at index `i * IMAGE_SIZE + j` in the flattened coordinate tensor. Double-check this by sampling a few known pixels.

### 1.4 — Create the PyTorch DataLoader

1. **Wrap in a `TensorDataset`.** Pass both the coordinate tensor `(65536, 2)` and the pixel tensor `(65536, 3)` to `torch.utils.data.TensorDataset`.

2. **Create the `DataLoader`.** Key settings:
   - `batch_size`: Set to `BATCH_SIZE` from config (8192 is a good starting point). This means each training step processes 8,192 pixels at a time.
   - `shuffle=True`: **Yes, shuffle.** Even though this is a single image, shuffling the pixel order prevents the network from overfitting to spatial locality patterns in the batch ordering.
   - `pin_memory=True`: If using GPU. This speeds up CPU→GPU data transfer.
   - `num_workers=0`: Keep at 0. The dataset is tiny and fits in memory; multiprocessing overhead would actually slow things down.

3. **Quick math check:** For a 256×256 image with batch size 8192, you get `65536 / 8192 = 8 batches per epoch`. For a 512×512 image, you get `262144 / 8192 = 32 batches per epoch`.

### 1.5 — Validation Sanity Check

Before moving on, write a quick test:

1. Take the first batch from the DataLoader.
2. Check that the coordinate tensor has shape `(BATCH_SIZE, 2)` and values in `[-1, 1]`.
3. Check that the pixel tensor has shape `(BATCH_SIZE, 3)` and values in `[-1, 1]`.
4. Pick a random coordinate from the batch, manually look up what pixel it should correspond to in the original image, and verify the color values match (within floating-point tolerance).

---

## Phase 2 — Architecture Construction (The SIREN)

You **cannot** use standard ReLU activations. ReLU networks suffer from _spectral bias_ — they learn low-frequency features first and struggle to ever capture high-frequency details like sharp edges. The SIREN architecture (Sitzmann et al., 2020) solves this by using periodic sine activations, enabling the network to represent signals across all frequencies simultaneously.

### 2.1 — The `SineLayer` Module

**File:** `src/model.py`

This is the fundamental building block. Every hidden layer in the SIREN is a `SineLayer`.

1. **Subclass `nn.Module`.**

2. **Constructor (`__init__`) parameters:**
   - `in_features` (int): Number of input features.
   - `out_features` (int): Number of output features.
   - `omega_0` (float): Frequency scaling factor. Default `30.0`.
   - `is_first` (bool): Whether this is the first layer of the network. Default `False`.

3. **Internal components:**
   - One `nn.Linear(in_features, out_features)` layer.
   - Store `omega_0` as an instance variable.

4. **Weight initialization** (called inside `__init__`, _after_ creating the `nn.Linear`):
   - Use `torch.no_grad()` context to modify weights in-place.
   - **If `is_first=True` (first layer):**
     - Initialize weights uniformly: `U(-1/in_features, 1/in_features)`.
     - In code: `self.linear.weight.uniform_(-1 / in_features, 1 / in_features)`.
   - **If `is_first=False` (hidden layers):**
     - Initialize weights uniformly: `U(-√(6/in_features) / omega_0, √(6/in_features) / omega_0)`.
     - In code: `self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, np.sqrt(6 / in_features) / omega_0)`.

   > **Why does this matter?** SIREN initialization is derived from maintaining unit variance of activations through the network. If you use PyTorch's default Kaiming initialization with sine activations, the signal will either explode or vanish by the second layer. The network will output noise and never converge. **Getting this wrong is the #1 cause of SIREN training failure.**

5. **Forward pass:**
   - Compute `self.linear(x)` to get the affine transformation.
   - Multiply by `self.omega_0`.
   - Apply `torch.sin()`.
   - Return the result.

   ```
   forward(x) = sin(omega_0 * Linear(x))
   ```

### 2.2 — The Full SIREN Model

1. **Constructor parameters:**
   - `in_features`: 2 (x and y coordinates).
   - `out_features`: 3 (R, G, B channels).
   - `hidden_features`: `HIDDEN_FEATURES` from config (256).
   - `hidden_layers`: `HIDDEN_LAYERS` from config (4).
   - `omega_0`: `OMEGA_0` from config (30.0).

2. **Build the network as an `nn.ModuleList`** (not a plain Python list — `nn.ModuleList` ensures PyTorch properly registers all parameters):

   - **Layer 0 (Input layer):** `SineLayer(in_features=2, out_features=256, omega_0=30.0, is_first=True)`
   - **Layers 1–3 (Hidden layers):** `SineLayer(in_features=256, out_features=256, omega_0=30.0, is_first=False)`
   - **Layer 4 (Output layer):** `nn.Linear(in_features=256, out_features=3)` — **NO sine activation**.

3. **Output layer initialization:**
   - The final `nn.Linear` layer should also be initialized with the SIREN-specific scheme for hidden layers (the `is_first=False` uniform distribution). This is because it receives sine-activated inputs.

4. **Forward pass:**
   - Pass the input through each `SineLayer` in sequence.
   - Pass through the final `nn.Linear`.
   - Optionally, apply `torch.tanh()` to clamp outputs to `[-1, 1]`. This is a design choice:
     - **With `tanh`:** Guarantees valid pixel range. Slightly slower convergence because gradients saturate at extremes.
     - **Without `tanh`:** Faster convergence, but some pixel values may slightly exceed `[-1, 1]`. You'll need to clamp when saving images.
   - **Recommendation:** Start _without_ `tanh`. Clamp to `[-1, 1]` only when converting to images for saving. This gives the network more gradient flow during training.

5. **Parameter count sanity check:** For a `[2 → 256 → 256 → 256 → 256 → 3]` network:
   - Layer 0: `(2 × 256) + 256 = 768` parameters
   - Layers 1–3: `(256 × 256) + 256 = 65,792` each → `65,792 × 3 = 197,376`
   - Output: `(256 × 3) + 3 = 771`
   - **Total: 198,915 parameters** (~200K). This is a tiny model. It should train in minutes on a GPU.

### 2.3 — Architecture Verification

Before proceeding, run this sanity check:

1. Instantiate the model.
2. Create a dummy input tensor of shape `(1, 2)` — one random `(x, y)` coordinate.
3. Pass it through the model.
4. Verify the output has shape `(1, 3)`.
5. Print the total parameter count: `sum(p.numel() for p in model.parameters())`.
6. Pass a batch of shape `(8192, 2)` and verify output is `(8192, 3)`.

---

## Phase 3 — The Baseline Run & State Capture

This phase establishes the "dense" (unpruned) baseline. Every pruning experiment is measured against this result. More importantly, you must capture the **exact initial weights** that the Lottery Ticket Hypothesis requires for weight rewinding.

### 3.1 — Initialize the Model and Save the "Ticket Stub"

**File:** `src/train.py`

1. **Set the random seed** by calling your `set_seed()` function.

2. **Instantiate the SIREN model** and move it to the configured device (GPU).

3. **🚨 CRITICAL STEP — Save the initial `state_dict` immediately:**

   ```python
   initial_state = model.state_dict()
   torch.save(initial_state, 'checkpoints/initial_weights.pth')
   ```

   > **Why is this the most important step in the entire project?**
   >
   > The Lottery Ticket Hypothesis by Frankle & Carlin (2018) states: a winning ticket is a subnetwork that, **when reset to its original initialization**, can match the performance of the full network. If you lose the initial weights, you cannot rewind. You cannot find lottery tickets. The entire experiment becomes meaningless.
   >
   > Save this file. Back it up. Never overwrite it.

4. **Also save the initial `state_dict` in memory** as a Python variable (e.g., `initial_state_dict = copy.deepcopy(model.state_dict())`). You will reference this many times during the pruning loops.

### 3.2 — Define the Loss Function and Optimizer

1. **Loss function:** `torch.nn.MSELoss()` — Mean Squared Error between predicted RGB values and ground truth RGB values. This is the standard choice for regression to continuous values.

2. **Optimizer:** `torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)`. Adam is the recommended optimizer for SIRENs. Do **not** use SGD — it converges far too slowly for this architecture.

3. **Learning rate scheduler (optional but recommended):** Use `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)`. This gradually decays the learning rate following a cosine curve, which helps the network settle into sharper representations during the final epochs.

### 3.3 — Implement the PSNR Metric

**File:** `src/utils.py`

Peak Signal-to-Noise Ratio (PSNR) is the standard metric for image reconstruction quality. It gives you a single number (in dB) that is much more interpretable than raw MSE.

1. **Formula:** `PSNR = 10 * log10(MAX² / MSE)`, where `MAX` is the maximum possible pixel value.

2. **In our case:** Since our pixel values are in `[-1.0, 1.0]`, the range is `2.0`, so `MAX = 2.0` and `MAX² = 4.0`.

   Alternatively, you can un-normalize to `[0, 1]` first and use `MAX = 1.0`.

3. **Implementation:**
   ```
   PSNR(mse) = 10 * log10(4.0 / mse)   # if targets are in [-1, 1]
   ```
   or
   ```
   PSNR(mse) = -10 * log10(mse)         # if targets are in [0, 1]
   ```

4. **Interpretation:**
   - **< 20 dB:** Very noticeable distortion. The image looks wrong.
   - **25–30 dB:** Decent reconstruction. Some blurring visible.
   - **30–35 dB:** Good reconstruction. Hard to tell from the original at a glance.
   - **35+ dB:** Excellent. Visually indistinguishable from the original.
   - **40+ dB:** Near-perfect. Pixel-level accuracy.

   Your dense baseline should achieve **at least 30 dB**, ideally 35+.

### 3.4 — Implement the `reconstruct_image` Helper

**File:** `src/utils.py`

You will call this function dozens of times throughout the experiment to visualize what the network "sees." Make it robust.

1. **Input:** The model and the full coordinate grid tensor.

2. **Process:**
   - Set the model to `model.eval()`.
   - Wrap in `torch.no_grad()` to save memory.
   - Pass the **entire** coordinate grid (all 65,536 coordinates at once) through the model. Since this is inference only, you don't need to batch it.
   - Get the output tensor of shape `(65536, 3)`.
   - Reshape to `(IMAGE_SIZE, IMAGE_SIZE, 3)`.
   - Un-normalize from `[-1, 1]` to `[0, 1]`: `output = (output + 1.0) / 2.0`.
   - Clamp to `[0, 1]` to handle any values slightly outside range.
   - Convert to `[0, 255]` uint8: `output = (output * 255).to(torch.uint8)`.
   - Convert to numpy array.
   - Set the model back to `model.train()`.

3. **Output:** A numpy array of shape `(IMAGE_SIZE, IMAGE_SIZE, 3)` with dtype `uint8`, ready to be saved with PIL or matplotlib.

4. **Optional:** Add a `save_path` parameter that, if provided, automatically saves the image using `PIL.Image.fromarray(array).save(save_path)`.

### 3.5 — The Training Loop

Write a clean, reusable training function that you will call for both the baseline and every pruning iteration.

**Function signature:**
```
train_model(model, dataloader, optimizer, loss_fn, epochs, device, coords_full, log_every=100)
```

**The loop structure:**

```
For each epoch from 1 to EPOCHS:
    For each batch (coords_batch, pixels_batch) in dataloader:
        1. Move both tensors to device.
        2. Forward pass: predicted_pixels = model(coords_batch)
        3. Compute loss: loss = loss_fn(predicted_pixels, pixels_batch)
        4. Zero gradients: optimizer.zero_grad()
        5. Backward pass: loss.backward()
        6. Update weights: optimizer.step()
        7. (Optional) Step the learning rate scheduler.

    If epoch % log_every == 0:
        Compute full-image MSE (using reconstruct_image or a separate eval pass).
        Compute PSNR from MSE.
        Print: "Epoch {epoch}/{EPOCHS}  |  Loss: {loss:.6f}  |  PSNR: {psnr:.2f} dB"
```

### 3.6 — Run the Baseline Training

1. Train the dense model for the full `EPOCHS` count.
2. Save the final reconstructed image to `outputs/baseline/dense_reconstruction.png`.
3. Save the trained dense model's `state_dict` to `checkpoints/dense_trained.pth` (useful for debugging later).
4. Record the final PSNR. This is your **upper bound** — no pruned network should exceed this.

### 3.7 — Visual Verification

Open `dense_reconstruction.png` and compare it side-by-side with the original image in `data/target_image.jpg`. They should be visually indistinguishable. If they're not:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Pure noise / static | SIREN initialization is wrong | Double-check the `is_first` flag and the uniform distribution formulas |
| Blurry, low-frequency only | `omega_0` is too small | Increase to 30 or even 60 |
| Unstable / oscillating loss | Learning rate too high | Reduce to `5e-5` |
| Slow convergence | Learning rate too low or too few epochs | Increase epochs to 2000+ or bump LR to `2e-4` |
| Color artifacts | Normalization mismatch | Verify both inputs and targets are `[-1, 1]` |

---

## Phase 4 — The Pruning Engine (The Logic Core)

This is the mathematical heart of the experiment. You will build three distinct pruning strategies, each answering a different question about the network's internal structure.

**File:** `src/pruning.py`

### 4.1 — Understanding PyTorch Pruning Internals

Before writing any code, you must understand exactly what `torch.nn.utils.prune` does under the hood, because you will need to manipulate its internals directly.

When you prune a layer, PyTorch does **not** delete weights. Instead, it:

1. **Renames** the `weight` parameter to `weight_orig`.
2. **Creates** a buffer called `weight_mask` — a binary tensor of the same shape where `1` means "keep" and `0` means "pruned."
3. **Registers a forward hook** that computes `weight = weight_orig * weight_mask` before every forward pass.
4. The "effective" weight is always `weight_orig * weight_mask`, but `weight_orig` still holds the original (or rewound) values.

This means:
- To access the actual underlying weights, use `layer.weight_orig`, not `layer.weight`.
- To see which weights are active, check `layer.weight_mask`.
- To change the underlying weights (e.g., for rewinding), modify `layer.weight_orig`.

### 4.2 — Helper: Collect All Prunable Layers

Write a helper function that iterates through the model and returns a list of `(module, "weight")` tuples for every `nn.Linear` layer. This list is what `prune.global_unstructured` expects.

```
def get_parameters_to_prune(model):
    Return a list of (module, 'weight') for every nn.Linear in the model.
    Use: [(module, 'weight') for module in model.modules() if isinstance(module, nn.Linear)]
```

> **Important:** This must include the final output layer. All linear layers participate in the global pruning pool.

### 4.3 — Helper: Calculate Current Sparsity

Write a function that calculates what percentage of the total weights are currently zero (pruned).

```
def compute_sparsity(model):
    total_params = 0
    pruned_params = 0
    For each nn.Linear in model:
        Count all elements in `.weight` (or `.weight_orig * .weight_mask` if pruned).
        Count zeros.
    Return pruned_params / total_params  (a float between 0.0 and 1.0)
```

### 4.4 — The "Winning Ticket" Pruning Function

This implements standard **Iterative Magnitude Pruning (IMP)** — the original Lottery Ticket Hypothesis method.

**Logic:** Remove the weights with the smallest absolute values. The intuition is that weights close to zero contribute almost nothing to the network's output, so removing them should cause minimal damage.

**Implementation:**

1. Call `prune.global_unstructured()` with:
   - `parameters`: The list from `get_parameters_to_prune(model)`.
   - `pruning_method`: `prune.L1Unstructured`.
   - `amount`: The `PRUNE_RATE` (e.g., `0.20` to prune 20% of **remaining** weights).

2. That's it. PyTorch handles mask creation and application automatically.

> **Key subtlety about `amount`:** When `amount` is a float, `global_unstructured` interprets it as "prune this fraction of the **currently unpruned** weights." So if you have 100 weights, prune 20%, you get 80 remaining. Next iteration, prune 20% of 80 = 64 remaining. This gives you exponential decay, which is exactly what IMP prescribes.

### 4.5 — The "Random Ticket" Pruning Function

This is your **control group**. It answers: "Is structure actually important, or is any sparse subnetwork equally good?"

**Logic:** Remove weights uniformly at random, regardless of their magnitude.

**Implementation:**

1. Call `prune.global_unstructured()` with:
   - `parameters`: The list from `get_parameters_to_prune(model)`.
   - `pruning_method`: `prune.RandomUnstructured`.
   - `amount`: The same `PRUNE_RATE` as the winning ticket.

### 4.6 — The "Losing Ticket" Pruning Function (Custom)

This is the **novel** part of the experiment. PyTorch does not provide a built-in method to prune the _largest_ weights. You must invert the logic.

**Logic:** Remove the weights with the _largest_ absolute values. These are the weights the network relied on _most_ during training. Forcing the network to work without them is like performing a targeted lobotomy on its most important neurons.

**Implementation — Step by step:**

1. **Collect all weight magnitudes.** Iterate through every `nn.Linear` layer. For each layer, get the absolute value of its effective weight tensor (if already pruned from a prior iteration, use `layer.weight_orig * layer.weight_mask`; if not yet pruned, use `layer.weight`). Flatten each into a 1D tensor and concatenate all of them into one single giant 1D tensor of absolute magnitudes.

2. **Determine the threshold.** You want to keep the _smallest_ weights and prune the _largest_. If `PRUNE_RATE = 0.20`:
   - Calculate the number of weights to **keep**: `n_keep = total_unpruned - n_to_prune`.
   - Sort the absolute magnitudes (ascending).
   - The threshold is the value at position `n_keep` — everything above this gets pruned.
   - Alternatively, use `torch.kthvalue()` or `torch.quantile()` for efficiency.

3. **Generate per-layer masks.** For each `nn.Linear` layer:
   - Compute the absolute weight tensor for that layer.
   - Create a binary mask: `mask = (abs_weight <= threshold).float()`.
   - This mask is `1` for small weights (keep) and `0` for large weights (prune).

4. **Apply the custom mask.** Use `prune.custom_from_mask(module, name='weight', mask=mask)` for each layer.

5. **Handle already-pruned layers.** If a layer has already been pruned in a prior iteration, it already has `weight_mask` and `weight_orig`. When you call `prune.custom_from_mask`, PyTorch will **multiply** the new mask with the existing mask. This is correct behavior — it accumulates pruning across iterations.

> **Edge case to handle:** When calculating magnitudes, make sure you only consider _currently unpruned_ weights (where the mask is 1). Already-pruned weights have been zeroed out and should not factor into the threshold calculation. Filter them out before sorting.

### 4.7 — Pruning Verification

After implementing each pruning function, verify it on a freshly instantiated model:

1. Instantiate a new SIREN model.
2. Print total parameter count.
3. Apply the pruning function with `amount=0.50` (prune 50%).
4. Call `compute_sparsity(model)` — it should report approximately 50%.
5. Verify that `weight_mask` exists on each `nn.Linear` layer.
6. Verify that approximately 50% of the effective weights (from `layer.weight`) are exactly zero.
7. Run a forward pass to make sure the model still produces output (not errors).

---

## Phase 5 — The Iterative Pruning Pipeline (The Experiment Loop)

This is the execution engine. You will run the **full pruning → rewind → retrain** loop three separate times — once for each ticket type (Winner, Random, Loser). Each run produces a series of reconstructed images at increasing sparsity levels and logs the corresponding PSNR values.

**File:** `src/pipeline.py`

### 5.1 — Define the Pruning Schedule

Before you start the loop, calculate exactly how many pruning iterations you need and what sparsity level each iteration corresponds to.

1. **Prune rate per iteration:** `p = PRUNE_RATE` (e.g., `0.20` = remove 20% of surviving weights each round).

2. **Remaining weights after `k` iterations:** `remaining(k) = (1 - p)^k`. This is exponential decay.

3. **Number of iterations to reach target sparsity:**
   ```
   k = ceil( log(TARGET_SPARSITY) / log(1 - p) )
   ```
   For `p = 0.20` and `TARGET_SPARSITY = 0.02`:
   ```
   k = ceil( log(0.02) / log(0.80) ) = ceil( -3.912 / -0.223 ) = ceil(17.53) = 18 iterations
   ```

4. **Pre-compute the sparsity schedule** as a list:
   ```
   Iteration 0:  100.0% remaining (dense baseline)
   Iteration 1:   80.0% remaining
   Iteration 2:   64.0% remaining
   Iteration 3:   51.2% remaining
   Iteration 4:   41.0% remaining
   Iteration 5:   32.8% remaining
   ...
   Iteration 15:   3.5% remaining
   Iteration 16:   2.8% remaining
   Iteration 17:   2.3% remaining
   Iteration 18:   1.8% remaining
   ```

   Store this schedule for logging purposes. It helps you label images and plot axes correctly.

### 5.2 — The Core Iterative Pruning Loop

This is the main function you will call three times. Write it as:

```
def run_pruning_pipeline(
    ticket_type: str,           # "winner", "random", or "loser"
    prune_function: callable,   # One of the three functions from Phase 4
    initial_weights_path: str,  # Path to initial_weights.pth
    config: dict,               # All hyperparameters
    dataloader,                 # The coordinate DataLoader
    coords_full,                # Full coordinate grid for reconstruction
):
```

**The loop — explained step by step:**

#### Step A — Initialize a Fresh Model

1. Create a brand-new SIREN model instance (identical architecture to the baseline).
2. Load the initial weights: `model.load_state_dict(torch.load(initial_weights_path))`.
3. Move to GPU.
4. This model starts in the exact same state as the baseline — completely untouched.

#### Step B — Run the Iteration Loop

```
For iteration in range(1, total_iterations + 1):
```

Inside each iteration you perform _five sub-steps_ in exact order:

##### Sub-step B.1 — Train the Current Model

1. Create a fresh optimizer: `Adam(model.parameters(), lr=LEARNING_RATE)`.
   - **You must create a new optimizer every iteration** because the parameter graph changes after pruning. Reusing an old optimizer will cause errors or momentum mismatches.
2. Create a fresh LR scheduler if using one.
3. Train for exactly `EPOCHS` epochs using the same training loop from Phase 3.
4. Record the final PSNR.

##### Sub-step B.2 — Evaluate and Save

1. Call `reconstruct_image()` to generate the network's current painting.
2. Save the image with a descriptive filename:
   ```
   outputs/{ticket_type}/iter_{iteration:02d}_remaining_{remaining_pct:.1f}pct.png
   ```
   For example: `outputs/winner/iter_05_remaining_32.8pct.png`
3. Log the result to `logs/experiment_log.csv`:
   ```csv
   ticket_type, iteration, remaining_pct, psnr_db
   winner, 5, 32.8, 31.45
   ```

##### Sub-step B.3 — Apply Pruning

1. Call your pruning function (Winner, Random, or Loser) on the trained model.
2. This creates/updates the `weight_mask` buffers on each layer.
3. After pruning, call `compute_sparsity()` to verify the actual sparsity matches your expected schedule.

##### Sub-step B.4 — Rewind Weights to Initialization

This is the **Lottery Ticket Hypothesis rewind step** and is the most technically delicate part of the entire project.

1. **Load the initial weights** from disk or from your in-memory copy:
   ```python
   initial_weights = torch.load(initial_weights_path)
   ```

2. **For each pruned layer in the model**, you must:
   - Access the current mask: `mask = layer.weight_mask`
   - Access the initial weight for this layer from the loaded state dict.
   - Overwrite the layer's `weight_orig` with the initial weight:
     ```python
     layer.weight_orig.data = initial_weights[layer_name + '.weight'] * mask
     ```
   - **Do the same for biases** if they exist:
     ```python
     layer.bias.data = initial_weights[layer_name + '.bias']
     ```
     (Biases are typically not pruned, so just copy them directly.)

3. **Why `weight_orig` and not `weight`?** Because after pruning, `weight` is a computed property (`weight_orig * weight_mask`). You must write to `weight_orig`.

4. **Why multiply by `mask`?** The initial weight values for pruned positions don't matter (they'll be zeroed by the mask during forward pass), but explicitly zeroing them keeps the state clean and prevents any numerical issues.

> **⚠️ Common mistake:** Forgetting to match the layer names between the saved `state_dict` keys and the current model's layer names. After pruning, PyTorch renames `weight` to `weight_orig`. So the key in `initial_weights` might be `net.0.linear.weight`, but in the pruned model, you need to access `net[0].linear.weight_orig`. Handle this name mapping carefully.

> **Alternative approach (simpler but equivalent):** Instead of manually iterating layers, you can:
> 1. Create a new unpruned model.
> 2. Load initial weights into it.
> 3. Copy the masks from the pruned model to the new model.
> 4. Apply `prune.custom_from_mask` for each layer using the saved masks.
>
> This is more code but less error-prone because you don't need to deal with name mismatches.

##### Sub-step B.5 — Loop Continuation Check

1. Call `compute_sparsity()` to get the current sparsity level.
2. If `remaining_percentage <= TARGET_SPARSITY`, break the loop.
3. Otherwise, continue to the next iteration (go back to B.1 — train again with the rewound, pruned weights).

### 5.3 — Running All Three Pipelines

Execute the pipeline three times:

```python
# 1. Winning Tickets (IMP — prune smallest)
run_pruning_pipeline("winner", prune_winning_ticket, ...)

# 2. Random Tickets (Random pruning — control group)
run_pruning_pipeline("random", prune_random_ticket, ...)

# 3. Losing Tickets (Reverse IMP — prune largest)
run_pruning_pipeline("loser", prune_losing_ticket, ...)
```

> **Time estimate:** For a 256×256 image with 1000 epochs and ~18 pruning iterations, each pipeline takes roughly 18,000 total training epochs. On a modern GPU (RTX 3060 or better), each pipeline should complete in **1–3 hours**. Total for all three: **3–9 hours**. Plan accordingly — start this overnight if needed.

> **Tip:** Run the Winning Ticket pipeline first. If the results look wrong (no clear winning ticket), there is a bug in your rewinding logic. Fix it before burning GPU hours on the other two pipelines.

### 5.4 — Intermediate Checkpoints (Insurance)

At the end of each pruning iteration, save:

1. The model's full state (`state_dict`, including masks): `checkpoints/{ticket_type}_iter_{k}.pth`.
2. The masks alone (for debugging): extract `weight_mask` from each layer and save as a dictionary.

This way, if training crashes at iteration 15 of 18, you can resume from the last checkpoint instead of starting over.

### 5.5 — Special Considerations for the Losing Ticket Pipeline

The Losing Ticket pipeline has unique challenges:

1. **Expect catastrophic collapse.** After removing the most important weights, the network will likely produce garbage. **This is expected and is the entire point.** The garbage _is_ the result.

2. **PSNR will plummet.** You may see PSNR drop below 10 dB. This is fine.

3. **Training may become unstable.** With only the weakest weights remaining, gradients can explode or vanish. If you see `NaN` losses:
   - Reduce the learning rate by 10× for this pipeline only (e.g., `1e-5`).
   - Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
   - If it still fails, that _is_ the result — the losing ticket cannot learn.

4. **The visual output is the star.** The abstract, glitchy, broken images that losing tickets produce are the most visually striking part of the experiment. Save them at high resolution.

---

## Phase 6 — Extraction, Visualization & Assembly

Once all three pipelines have finished, you process and visualize the results to tell the story.

**File:** `src/visualize.py`

### 6.1 — PSNR Convergence Curves

Create a line chart that mathematically proves the Lottery Ticket Hypothesis holds for INRs.

1. **Load `experiment_log.csv`** into a dictionary or pandas DataFrame.

2. **X-axis:** "Remaining Weights (%)" — decreasing from 100% to ~2%. Use a logarithmic scale for the x-axis because the sparsity increases exponentially.

3. **Y-axis:** "PSNR (dB)" — the reconstruction quality metric.

4. **Plot three lines:**
   - 🟢 **Winning Ticket** (green, solid line): Should stay high (near the dense baseline) for a remarkably long time before eventually declining.
   - 🔵 **Random Ticket** (blue, dashed line): Should decline steadily and monotonically.
   - 🔴 **Losing Ticket** (red, dotted line): Should crash early and dramatically.

5. **Add a horizontal reference line** at the dense baseline PSNR level, labeled "Dense Baseline."

6. **Add a vertical reference line** at the "critical sparsity" — the point where the Winning Ticket finally starts to diverge from the baseline. This is a key finding of the experiment.

7. **Styling:**
   - Title: "Lottery Tickets in a SIREN: PSNR vs. Network Sparsity"
   - Legend: Clear labels for all three lines.
   - Grid: Light grid for readability.
   - Font: Use a clean sans-serif font (e.g., "DejaVu Sans" or similar).

8. **Save to:** `outputs/plots/psnr_vs_sparsity.png` at 300 DPI.

### 6.2 — The Degradation Timeline (Per-Ticket-Type Image Grids)

For each ticket type, create a horizontal strip showing how the image degrades as you remove more weights.

1. **Select key sparsity checkpoints.** Don't show all 18 iterations — pick 6–8 visually meaningful ones:
   ```
   100%, 80%, 50%, 20%, 10%, 5%, 3%, 2%
   ```

2. **Load the saved images** from the corresponding `outputs/{ticket_type}/` directory.

3. **Stitch them horizontally** into a single wide image. Use matplotlib's `subplot` or PIL's `Image.new` to create a grid.

4. **Label each sub-image** with its remaining weight percentage. Use small, clean text overlaid on or above each image.

5. **Create three grids** — one for Winners, one for Random, one for Losers.

6. **Save to:**
   - `outputs/plots/timeline_winner.png`
   - `outputs/plots/timeline_random.png`
   - `outputs/plots/timeline_loser.png`

### 6.3 — The Side-by-Side "Showdown" Grid

This is the **hero image** of the entire project — the image that will make people stop scrolling.

1. **Pick one extreme sparsity level** where the differences are most dramatic (e.g., 5% or 3% remaining weights).

2. **Load the three images** (Winner, Random, Loser) at that exact sparsity level.

3. **Create a 1×4 grid** (or 2×2):
   - **Column 1:** The original target image, labeled "Original."
   - **Column 2:** The Winning Ticket reconstruction, labeled "Winning Ticket (5%)."
   - **Column 3:** The Random Ticket reconstruction, labeled "Random Ticket (5%)."
   - **Column 4:** The Losing Ticket reconstruction, labeled "Losing Ticket (5%)."

4. **Styling:** Use a dark background (#1a1a2e or similar) with white text labels. Add thin white borders between images. This dark theme makes the images pop and looks professional.

5. **Save to:** `outputs/plots/showdown_grid.png` at high resolution (at least 2000px wide).

### 6.4 — The Degradation Animation (GIF / MP4)

Create a smooth animation showing the image dissolving as the network is pruned.

1. **Collect all saved images** for a given ticket type, ordered by decreasing remaining weight percentage.

2. **Using `imageio`:**
   ```python
   images = [imageio.imread(path) for path in sorted_image_paths]
   imageio.mimsave('outputs/plots/degradation_winner.gif', images, duration=0.5)
   ```

3. **Duration per frame:** 0.3–0.5 seconds. Slow enough to appreciate each step, fast enough to see the progression.

4. **Create three GIFs:**
   - `outputs/plots/degradation_winner.gif`
   - `outputs/plots/degradation_random.gif`
   - `outputs/plots/degradation_loser.gif`

5. **Optional enhancement:** Add a frame counter or sparsity percentage as text burned into each frame using PIL's `ImageDraw`.

### 6.5 — Difference Maps (Optional but Impressive)

Show _where_ the network fails by visualizing the pixel-by-pixel error.

1. **For a given reconstruction:** Compute `|original - reconstruction|` per pixel.
2. **Normalize** this difference to `[0, 255]` for visibility.
3. **Apply a colormap** (e.g., matplotlib's `'hot'` or `'inferno'`) to make errors pop.
4. **Interpretation:** Bright areas = large error = the network can't reconstruct this region. Dark areas = accurate.
5. This shows whether the network loses high-frequency details first (edges, textures) or low-frequency ones (broad color regions).

### 6.6 — Spectral Analysis (Optional, Advanced)

For readers who want deeper insight into _why_ the images degrade the way they do:

1. **Compute the 2D Fourier Transform** of each reconstruction using `torch.fft.fft2`.
2. **Compute the magnitude spectrum** and shift zero-frequency to center with `torch.fft.fftshift`.
3. **Visualize** the frequency spectrum as a log-scale heatmap.
4. **Compare spectra** at different sparsity levels: The Winning Ticket should preserve high-frequency content longer. The Losing Ticket should lose high frequencies first and fastest.

### 6.7 — Final Summary Table

Generate a clean markdown or CSV table summarizing the key results at selected sparsity levels:

| Remaining % | Winner PSNR (dB) | Random PSNR (dB) | Loser PSNR (dB) |
|---|---|---|---|
| 100% (Dense) | 38.2 | 38.2 | 38.2 |
| 50% | 37.8 | 33.1 | 18.5 |
| 20% | 35.2 | 26.7 | 12.3 |
| 10% | 31.1 | 20.4 | 9.8 |
| 5% | 27.5 | 15.2 | 8.1 |
| 3% | 22.3 | 12.8 | 7.2 |

*(These are illustrative example values — your actual results will differ.)*

---

## Appendix A — Common Pitfalls & Debugging Guide

| # | Problem | Symptom | Root Cause | Solution |
|---|---------|---------|-----------|----------|
| 1 | SIREN outputs pure noise | Black/white static, ~0 dB PSNR | Wrong weight initialization | Verify `is_first` flag; check uniform distribution bounds |
| 2 | Model trains but image is blurry | Low PSNR (< 25 dB), only broad color | `omega_0` too low | Increase to 30 or 60 |
| 3 | Loss oscillates wildly | Loss jumps up and down | Learning rate too high | Reduce LR to 5e-5 |
| 4 | PSNR identical for all ticket types | No difference between Winner/Random/Loser | Rewinding is broken — not loading initial weights | Verify `weight_orig` is being overwritten correctly |
| 5 | Pruning percentages are wrong | Sparsity doesn't match expected schedule | Not accounting for already-pruned weights | Use `amount` as fraction, not absolute count |
| 6 | `KeyError` during weight rewind | State dict key mismatch | Pruning renames `weight` to `weight_orig` | Map keys correctly between initial and pruned model |
| 7 | `NaN` loss during Loser training | Loss becomes `nan` after a few epochs | Gradient explosion with only weak weights | Add gradient clipping; reduce LR |
| 8 | Out of GPU memory | CUDA OOM error | Batch size too large for GPU | Reduce `BATCH_SIZE` to 4096 or smaller |
| 9 | Images look transposed/flipped | Reconstruction is rotated 90° | Wrong `meshgrid` indexing | Use `indexing='ij'` and verify coordinate order |
| 10 | All reconstructions look identical | Pruning has no effect | Masks not being applied | Check that `weight_mask` is registered on each layer |

---

## Appendix B — Expected Timeline

| Phase | Estimated Time | Notes |
|---|---|---|
| Phase 0 (Setup) | 30 minutes | Environment, directories, config |
| Phase 1 (Data Pipeline) | 1 hour | Dataset class + verification |
| Phase 2 (SIREN) | 1–2 hours | Architecture + initialization debugging |
| Phase 3 (Baseline) | 1 hour code + 10 min GPU | Training loop + verification |
| Phase 4 (Pruning Engine) | 2–3 hours | Custom losing ticket pruning is the hardest part |
| Phase 5 (Pipeline) | 2 hours code + 3–9 hours GPU | Three full pipelines |
| Phase 6 (Visualization) | 2–3 hours | Plots, grids, GIFs |
| **Total** | **~10–20 hours** | **Over 2–3 days** |

---

## Appendix C — Key References

1. **Sitzmann, V., Martel, J.N.P., Bergman, A.W., Lindell, D.B., & Wetzstein, G.** (2020). *Implicit Neural Representations with Periodic Activation Functions.* NeurIPS 2020. — The paper that introduced SIREN.

2. **Frankle, J. & Carlin, M.** (2018). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR 2019. — The foundational LTH paper.

3. **Frankle, J., Dziugaite, G.K., Roy, D.M., & Carlin, M.** (2020). *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML 2020. — Introduces "late rewinding" as an improvement to the original LTH.

4. **PyTorch Pruning Tutorial.** [https://pytorch.org/tutorials/intermediate/pruning_tutorial.html](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) — Official guide to `torch.nn.utils.prune`.

---

> **End of Implementation Plan.** Follow each phase sequentially. Do not skip ahead. Every step exists for a reason. Good luck — the images you produce will be unlike anything you've seen before.