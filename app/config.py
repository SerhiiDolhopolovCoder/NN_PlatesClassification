import os

import torch


NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

TRAIN_DATA_PATH = "../data/train"
VALID_DATA_PATH = "../data/valid"

device = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)