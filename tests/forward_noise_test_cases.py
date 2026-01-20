import unittest
import numpy as np
import torch


from tests.matplotlib_test_utils import show_plot_non_blocking
from src.utils.some_other_name import plot_image

class MyTestCase(unittest.TestCase):
    def test_plot_image_works_with_np_vector(self):
        # Example: your image
        image = np.random.rand(3, 224, 224)  # (C, H, W)

        plot_image(image)
        show_plot_non_blocking()

    def test_plot_image_works_with_torch_vector(self):
        from src.utils.some_other_name import plot_image
        # Example: your image
        image = torch.rand(3, 224, 224)  # (C, H, W)

        plot_image(image)
        show_plot_non_blocking()


if __name__ == '__main__':
    unittest.main()
