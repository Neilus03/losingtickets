import os
import torch

IMAGE_PATH = os.path.join("data", "target_image.jpg")
IMAGE_SIZE = 256
HIDDEN_FEATURES = 256
HIDDEN_LAYERS = 4
OMEGA_0 = 30.0
LEARNING_RATE = 1e-4
EPOCHS = 1000
BATCH_SIZE = 8192
PRUNE_RATE = 0.20
TARGET_SPARSITY = 0.02
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
