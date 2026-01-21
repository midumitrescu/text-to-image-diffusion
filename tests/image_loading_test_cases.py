import os
import unittest

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.image_utils import load_image, show_image, ensure_correct_dimensions

BASE_DIR = Path(__file__).resolve().parent
images_folder_for_tests = os.path.join(BASE_DIR, "data")

from numpy.testing import assert_array_equal


def load_images_for_test(file_id: str):
    return load_image(file_id, images_folder=images_folder_for_tests)


class ImageLoadingTestCases(unittest.TestCase):

    def test_image_1_loads_correctly(self):
        object_under_test = load_images_for_test("000001.jpg")
        self.assertEqual(object_under_test.shape, (3, 218, 178))

    def test_image_2_loads_correctly(self):
        object_under_test = load_images_for_test("000002.jpg")
        self.assertEqual(object_under_test.shape, (3, 218, 178))

    @staticmethod
    def test_plot_image_1_exploratory():
        object_under_test = load_images_for_test("000001.jpg")
        show_image(object_under_test)
        plt.close("all")

    def test_make_sure_T_operator_reversed_before_show(self):
        image_name = "000001.jpg"
        img = Image.open(f"{BASE_DIR}/../img_align_celeba/img_align_celeba/{image_name}")
        original_array = np.array(img)

        object_under_test = load_image(image_name)

        self.assertEqual(original_array.shape, (218, 178, 3))
        self.assertEqual(object_under_test.shape, (3, 218, 178))
        assert_array_equal(ensure_correct_dimensions(object_under_test), original_array)

    @staticmethod
    def test_image_loads_correctly():
        for weird_file in ["121959.jpg", "053492.jpg", "106818.jpg"]:
            img = Image.open(f"{BASE_DIR}/../img_align_celeba/img_align_celeba/{weird_file}")
            plt.imshow(img)
            plt.show()
            plt.close("all")

            object_under_test = load_images_for_test(weird_file)
            show_image(object_under_test)


if __name__ == '__main__':
    unittest.main()
