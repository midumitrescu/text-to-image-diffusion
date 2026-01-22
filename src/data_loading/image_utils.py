from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2

BASE_DIR = Path(__file__).resolve().parent
default_images_folder = BASE_DIR.joinpath("..", "..", "img_align_celeba", "img_align_celeba")


def ensure_correct_dimensions(img_array):
    img = np.array(img_array)
    if img.ndim == 3:
        if (img.shape[-1] != 3):
            img = np.transpose(img, [1, 2, 0])
    return img


def load_image(file_id: str, images_folder=default_images_folder, should_transpose = True) -> np.ndarray:
    img = Image.open(f"{images_folder}/{file_id}").convert("RGB")
    img = ensure_correct_dimensions(img)
    np_array = np.array(img)
    if not should_transpose:
        return np_array
    return np_array.transpose(2, 0, 1)


def show_image(img_array) -> None:
    img = ensure_correct_dimensions(img_array)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


def transformer_from_rgb_format(img_size):
    height, width = get_image_size(img_size)
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
        ToTensorV2()
    ])


def get_image_size(img_size):
    if type(img_size) is tuple:
        height, width = img_size
    else:
        height, width = img_size, img_size
    return height, width


def from_rgb_format(image, img_size):
    transform = transformer_from_rgb_format(img_size=img_size)
    return transform(image=image)["image"].float()

def to_rgb_format(x_hat):
    """
    x_hat: torch tensor (C,H,W) or (B,C,H,W), float32 [-1,1]
    Returns: numpy array uint8, HWC or BHWC [0,255]
    """
    # if batched, loop over batch
    if x_hat.ndim == 4:
        x_hat = x_hat.detach()
        imgs = []
        for xi in x_hat:
            img = ((xi.clamp(-1,1) + 1)/2 * 255).round().byte()
            img = img.permute(1,2,0).numpy()  # CHW -> HWC
            imgs.append(img)
        return np.stack(imgs)
    else:
        x_hat = x_hat.detach()
        return ((x_hat.clamp(-1,1)+1)/2 * 255).round().byte()
