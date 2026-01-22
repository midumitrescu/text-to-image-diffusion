import unittest

from data_loading.image_utils import transformer_from_rgb_format, to_rgb_format, show_image, from_rgb_format
from tests.image_loading_test_cases import load_images_for_test

test_image = load_images_for_test("000001.jpg")
image = test_image.transpose(1, 2, 0)

target_image_size = (128, 128)
transform = transformer_from_rgb_format(target_image_size)
boxed_image = transform(image=image)["image"].float()


class VAEUtilsTestCase(unittest.TestCase):

    def test_transformations_produces_desired_shape(self):
        self.assertEqual((3, 218, 178), test_image.shape)
        self.assertEqual((3, 128, 128), boxed_image.shape)

    def test_transformation_produces_values_between_minus_1_to_1(self):
        self.assertGreaterEqual(1, boxed_image.max().numpy())
        self.assertLessEqual(-1, boxed_image.min().numpy())


    def test_transformations_are_deterministic(self):
        image = load_images_for_test("000001.jpg")
        test_image = image.transpose(1, 2, 0)
        boxed_image = from_rgb_format(test_image, img_size=(128, 128))

        self.assertEqual((218, 178, 3), test_image.shape)
        self.assertEqual((3, 128, 128), boxed_image.shape)
        transformed_back = to_rgb_format(boxed_image)
        show_image(boxed_image)
        show_image(transformed_back)
        self.assertGreaterEqual(255, transformed_back.max().numpy())
        self.assertLessEqual(0, transformed_back.min().numpy())


