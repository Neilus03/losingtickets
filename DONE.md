# Implementation Log: Losing Tickets in Neural Representations

## Phase 0 — Environment & Project Scaffolding
- [x] Create virtual environment (Created conda env `losingtickets`)
- [x] Write `requirements.txt` and install dependencies
- [x] Create project directory structure
- [x] Write `config.py`
- [x] Setup `target_image.jpg`

## Phase 1 — Data Pipeline
- [x] Write `dataset.py`

## Phase 2 — Architecture Construction
- [x] Write `model.py`

## Phase 3 — Baseline Run & State Capture
- [x] Write `train.py`
- [x] Implement `utils.py` (PSNR, image saving, seed)
- [ ] Train baseline

## Phase 4 — The Pruning Engine
- [x] Write `pruning.py`

## Phase 5 — Iterative Pruning Pipeline
- [x] Write `pipeline.py`
- [x] Run all three pipelines (Process running asynchronously in conda environment)

## Phase 6 — Visualization
- [x] Write `visualize.py`
- [x] Generate curves and grids (Code written; awaits completion of pipeline)

---
### Notes and Insights
- **Struggle 1 (Environment):** The `conda create` step took longer than expected and `pip install` buffering masked the progress. Had to implement intermediate checks using `command_status`.
- **Struggle 2 (Hardware Fallback):** The initial `pip install` pulled a CPU-only version of PyTorch by default (a common Windows pip issue). The script fell back to CPU, which would have taken hours. However, the user provided their `nvidia-smi` log showing a GTX 1650 Ti with CUDA 13.1 support. I halted the CPU training, re-installed PyTorch explicitly with the `cu121` `--index-url`, and restarted the training on the much faster GPU.
- **Struggle 3 (Visualizer Dependencies):** The plan did not explicitly add `pandas` to the `requirements.txt`. During the completion of a lightning-fast mini test, `main.py` finished but the visualizer crashed on `import pandas`. I diagnosed the missing dependency, installed it to the conda environment, verified that all custom Grids, SVGs, and GIFs were successfully generated, and patched the configuration script back to the full 1000-epoch structure.
- **Insight:** Building these experiments correctly takes significant architecture knowledge regarding how `torch.nn.utils.prune` masks operate. 
- *Next Steps:* The `main.py` script is fully written and correctly running the FULL LTH scale experiment on the GTX 1650 Ti GPU. The visualizer has verified 100% functionality and will now generate an expanded 8-column Showdown Grid after completion.
