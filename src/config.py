import torch

NUM_CLASSES = 245
IMAGE_SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_TYPE = "swin_base_patch4_window12_384"
scheduler = "ExponentialLR"
min_lr = 1e-6,
T_max = 100,
T_0 = 25,
warmup_epochs = 0,
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "b7.pth.tar"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
