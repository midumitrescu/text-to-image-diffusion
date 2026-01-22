import unittest

import numpy as np
import torch
from data_loading.image_utils import show_image

from tests.matplotlib_test_utils import show_plot_non_blocking


class GaussianNoiseGenerationTestCase(unittest.TestCase):
    def test_plot_image_works_with_np_vector(self):
        image = np.random.rand(3, 224, 224)  # (C, H, W)

        show_image(image)
        show_plot_non_blocking()

    def test_plot_image_works_with_torch_vector(self):
        image = torch.rand(3, 224, 224)  # (C, H, W)

        show_image(image)
        show_plot_non_blocking()


if __name__ == '__main__':
    unittest.main()
