# 🎟️ Losing Tickets in Neural Representations
### Exploring the Lottery Ticket Hypothesis in SIRENs

This repository contains the code and experiments for the **Losing Tickets in Neural Representations** project, which investigates the Lottery Ticket Hypothesis (LTH) within Sinusoidal Representation Networks (SIRENs).

## 📖 [Read the Full Interactive Blog Post](https://neilus03.github.io/losingtickets/)
*Visual diagnostics, interactive PSNR curves, and detailed experimental analysis.*

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Neilus03/losingtickets.git
cd losingtickets
```

### 2. Install dependencies
```bash
pip install torch torchvision numpy pillow tqdm
```

### 3. Run the experiment
```bash
python main.py
```
This will:
- Train a baseline dense SIREN model on the Mona Lisa.
- Run the iterative pruning loop for **Winner**, **Random**, and **Loser** tickets.
- Generate reconstruction images and PSNR logs in the `outputs/` directory.

## 📁 Project Structure
- `src/`: Core implementation (SIREN, pruning logic, training loop).
- `docs/`: GitHub Pages code and interactive blog.
- `checkpoints/`: Model state dicts for different pruning iterations.
- `outputs/`: Reconstructed images and benchmark results.

---
**Author:** [Neil De La Fuente](https://www.linkedin.com/in/neil-de-la-fuente/)
