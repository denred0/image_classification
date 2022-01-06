import shutil
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path

from my_utils import recreate_folders, seed_everything, get_all_files_in_folder

import config


def train_val_test_split(input_data_dir: str, output_data_dir: str, val_fraction=0.1, test_fraction=0.1) -> None:
    seed_everything(config.SEED)

    recreate_folders(Path(output_data_dir), ["train", "test", "val"])

    train_val_test_dict = get_train_val_test_images(input_data_dir, val_fraction, test_fraction)

    copy_images_to_folder_classes(train_val_test_dict)


def get_train_val_test_images(path_to_dataset: str, val_fraction=0.1, test_fraction=0.1) -> dict:
    path_to_dataset = Path(path_to_dataset)

    all_images = get_all_files_in_folder(Path(path_to_dataset), ["*." + config.IMAGE_EXT])
    all_images_labels = [str(path).split(os.sep)[-2] for path in all_images]
    train_files, val_test_files = train_test_split(all_images, test_size=(val_fraction + test_fraction),
                                                   stratify=all_images_labels)

    test_fraction_from_val = test_fraction / (val_fraction + test_fraction)

    val_test_labels = [str(path).split(os.sep)[-2] for path in val_test_files]
    val_files, test_files = train_test_split(val_test_files, test_size=test_fraction_from_val,
                                             stratify=val_test_labels)

    return {"train": train_files, "val": val_files, "test": test_files}


def copy_images_to_folder_classes(train_val_test_dict: dict) -> None:
    for part, files in train_val_test_dict.items():
        labels = [str(path).split(os.sep)[-2] for path in files]
        for label in set(labels):
            Path(output_data_dir).joinpath(part).joinpath(label).mkdir(parents=True, exist_ok=True)

        for file, label in tqdm(zip(files, labels), desc=part + " preparing", total=len(labels)):
            shutil.copy(file, Path(output_data_dir).joinpath(part).joinpath(label))


if __name__ == "__main__":
    input_data_dir = "data/dataset"
    output_data_dir = "data/train_val_test_split"
    val_fraction = 0.1
    test_fraction = 0.1
    train_val_test_split(input_data_dir, output_data_dir, val_fraction, test_fraction)
