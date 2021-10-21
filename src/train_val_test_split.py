import numpy as np
import pickle
import shutil
# import os
# import pandas as pd
from tqdm import tqdm

# pytorch related imports
# from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from src.dataset import ICPDataset
#
# import pytorch_lightning as pl
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
# from collections import Counter

from pathlib import Path


def train_val_test_split(data_dir):
    output_dir = Path("data/train_val_test_split")
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    path = Path(data_dir)

    train_val_files = list(path.rglob('*.jpg'))
    train_val_labels = [path.parent.name for path in train_val_files]

    train_files, val_test_files = train_test_split(train_val_files, test_size=0.2, stratify=train_val_labels)

    train_labels = [path.parent.name for path in train_files]

    for label in set(train_labels):
        Path(output_dir).joinpath("train").joinpath(label).mkdir(parents=True, exist_ok=True)

    for file, label in tqdm(zip(train_files, train_labels), desc="Train preparing", total=len(train_labels)):
        shutil.copy(file, Path(output_dir).joinpath("train").joinpath(label))

    val_test_labels = [path.parent.name for path in val_test_files]

    only_train_and_val = True

    if only_train_and_val:
        for label in set(val_test_labels):
            Path(output_dir).joinpath("val").joinpath(label).mkdir(parents=True, exist_ok=True)

        for file, label in tqdm(zip(val_test_files, val_test_labels), desc="Validation preparing", total=len(val_test_labels)):
            shutil.copy(file, Path(output_dir).joinpath("val").joinpath(label))
    else:

        val_files, test_files = train_test_split(val_test_files, test_size=0.5, stratify=val_test_labels)

        val_labels = [path.parent.name for path in val_files]
        test_labels = [path.parent.name for path in test_files]

        for label in set(val_labels):
            Path(output_dir).joinpath("val").joinpath(label).mkdir(parents=True, exist_ok=True)

        for file, label in tqdm(zip(val_files, val_labels), desc="Validation preparing", total=len(val_labels)):
            shutil.copy(file, Path(output_dir).joinpath("val").joinpath(label))

        for label in set(test_labels):
            Path(output_dir).joinpath("test").joinpath(label).mkdir(parents=True, exist_ok=True)

        for file, label in tqdm(zip(test_files, test_labels), desc="Test preparing", total=len(test_labels)):
            shutil.copy(file, Path(output_dir).joinpath("test").joinpath(label))


if __name__ == "__main__":
    train_val_test_split(data_dir="data/dataset (copy)")
