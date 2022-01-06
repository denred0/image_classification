import torch

NUM_CLASSES = 3
IMAGE_SIZE = 456
IMAGE_EXT = "png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_TYPE = "tf_efficientnet_b5"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 4

scheduler = "ExponentialLR"
min_lr = 1e-6,
T_max = 100,
T_0 = 25,
warmup_epochs = 0,
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
WEIGHT_DECAY = 1e-4

SEED = 42
