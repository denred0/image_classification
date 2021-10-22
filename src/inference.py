import timm
import torch
import albumentations as A
import cv2
import shutil
import os
import pickle
import numpy as np

from tqdm import tqdm
from os import walk
from pathlib import Path

from torch import nn, optim
from albumentations.pytorch import ToTensorV2

import config


def get_model(model_type, num_classes):
    model = timm.create_model(model_type, pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model


val_transforms = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.Normalize(mean=config.MEAN, std=config.STD),
    ToTensorV2()
], p=1.0)

model = get_model(config.MODEL_TYPE, config.NUM_CLASSES)
model.load_state_dict(torch.load("logs/exp_1/acc_0.7630_epoch_1.pth"))
model = model.to(config.DEVICE)
model.eval()
# print(model)

class_names = []
with open("classes.txt") as file:
    lines = file.readlines()
    class_names = [line.rstrip() for line in lines]

output_dir = Path('data/inference/output_dishes')
if output_dir.exists() and output_dir.is_dir():
    shutil.rmtree(output_dir)
Path(output_dir).mkdir(parents=True, exist_ok=True)

label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

correct = 0
all = 0
sm = torch.nn.Softmax()

input_dir = "data/inference/input_dishes"
for root, dirs, files in os.walk(input_dir):
    for folder in tqdm(dirs):

        p = os.path.join(input_dir, folder) + os.path.sep

        _, _, images_list = next(walk(p))

        for img_name in images_list:

            img = cv2.imread(os.path.join(p, img_name), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            augmented = val_transforms(image=img)
            img_array = augmented['image'].unsqueeze(0)

            y_hat = model(img_array.to(config.DEVICE))

            probabilities = sm(y_hat)
            predicted_class = torch.argmax(probabilities, dim=1).cpu().detach().numpy()[0]
            class_name = class_names[predicted_class]

            folder_name = ""
            if class_name == folder:
                correct += 1
                folder_name = "_" + folder
            else:
                folder_name = folder + "  -->  " + class_name

            folder_result = Path(output_dir).joinpath(folder_name)
            if not folder_result.exists():
                Path(folder_result).mkdir(parents=True, exist_ok=True)

            shutil.copy(os.path.join(p, img_name), folder_result)

            all += 1

print(f"Accuracy: {round(correct / all * 100, 2)}%")
