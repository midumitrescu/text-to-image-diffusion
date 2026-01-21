import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
default_images_folder = BASE_DIR.joinpath("..", "..", "img_align_celeba", "img_align_celeba")

def ensure_correct_dimensions(img_array):
    img = np.array(img_array)
    if img.ndim == 3:
        if (img.shape[-1] != 3):
            img = np.transpose(img, [1, 2, 0])
    return img


def load_image(file_id: str, images_folder=default_images_folder) -> np.ndarray:
    img = Image.open(f"{images_folder}/{file_id}").convert("RGB")
    img = ensure_correct_dimensions(img)
    np_array = np.array(img)
    return np_array.transpose(2, 0, 1)


def show_image(img_array) -> None:
    img = ensure_correct_dimensions(img_array)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


