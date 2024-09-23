import torch

# Hyperparameter Tuning

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 8
# NUM_WORKERS = 20
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True
