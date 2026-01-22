import unittest

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from data_loading.image_utils import load_image, show_image, transformer_from_rgb_format, to_rgb_format, from_rgb_format
from vae.api import app
from vae.vae_utils import load_vae

client = TestClient(app)

examples_folder = Path(__file__).resolve().parent.joinpath("..", "data")
example_file = examples_folder.joinpath( "106818.jpg")

class VAEApiTestCase(unittest.TestCase):

    def test_coding_and_decoding_works(self):
        vae = load_vae(model_file=None, device="cpu")
        test_image = load_image("000001.jpg", should_transpose=False)
        boxed_image = from_rgb_format(test_image, img_size=(128, 128))
        posterior = vae.encode(boxed_image.unsqueeze(0))
        z = posterior.latent_dist.mean
        x_hat = vae.decode(z).sample.squeeze(0)

        self.assertEqual((1, 4, 16, 16), z.shape)
        self.assertEqual((3, 128, 128), x_hat.shape)
        show_image(test_image)
        show_image(to_rgb_format(x_hat))





if __name__ == '__main__':
    unittest.main()
