import random
import os
import torch
import numpy as np
import timm
import shutil
import torch.nn as nn

from pathlib import Path
from typing import List


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model(model_type, num_classes):
    model = timm.create_model(model_type, pretrained=True)
    # in_features = model.head.in_features
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def recreate_folders(source_dir: Path, folders_list: List) -> None:
    for directory in folders_list:
        output_dir = source_dir.joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed