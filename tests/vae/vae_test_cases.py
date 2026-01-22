import unittest

from data_loading.image_utils import show_image, to_rgb_format, from_rgb_format
from fastapi.testclient import TestClient
from vae.api import app
from vae.vae_utils import load_vae

from tests.image_loading_test_cases import load_images_for_test

client = TestClient(app)

class VAEApiTestCase(unittest.TestCase):

    def test_coding_and_decoding_works(self):
        vae = load_vae(model_file=None, device="cpu")
        test_image = load_images_for_test("000001.jpg").transpose(1, 2, 0)
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
