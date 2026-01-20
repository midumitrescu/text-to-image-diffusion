import numpy as np
import matplotlib.pyplot as plt
import torch

__test__ = False

def plot_image(image_array: torch.Tensor):
    # Convert to (H, W, C)
    image = np.transpose(image_array, (1, 2, 0))

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.close()