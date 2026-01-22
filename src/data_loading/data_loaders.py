import albumentations as A
from albumentations.pytorch import ToTensorV2
from diffusers import AutoencoderKL
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.image as mpimg
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import make_image_grid
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from pathlib import Path
import random

from data_loading.image_utils import transformer_from_rgb_format

default_images_dir = "celeba-dataset-short/img_align_celeba"
default_label_file = "text_labels.json"

BASE_DIR = Path(__file__).resolve().parent

class AugmentedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, center_mean=True, text_labels=True,
                 return_filename_as_target=False, label_file=default_label_file):
        self.center_mean = center_mean
        self.return_filename_as_target = return_filename_as_target
        self.text_labels = json.load(open(BASE_DIR.joinpath(label_file))) if text_labels else {}
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        file_id = Path(path).name

        if self.text_labels:
            text_labels = self.text_labels.get(file_id, [""])
            text_labels = [t for t in text_labels if len(t.split()) <= 7]
            target = random.choice(text_labels)
        if self.transform is not None:
            image_np = A.Compose([A.ToFloat(max_value=255)])(image=np.array(image))['image']
            augmented = self.transform(image=image_np)
            image = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.center_mean:
            image = image * 2 - 1

        target = path if self.return_filename_as_target else target

        return image, target


def load_data(img_size, batch_size,  center_mean=True, text_labels=False,
              train_ratio=1.0, get_filepaths=False, images_dir = default_images_dir):
    transform = transformer_from_rgb_format(img_size)
    dataset = AugmentedImageFolder(root=images_dir, transform=transform, center_mean=center_mean,
                                         text_labels=text_labels, return_filename_as_target=get_filepaths)

    number_images = len(dataset)
    test_size = int((1 - train_ratio) * len(dataset))
    train_size = number_images - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader