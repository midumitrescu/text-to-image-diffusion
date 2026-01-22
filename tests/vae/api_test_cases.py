import io
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from data_loading.image_utils import from_rgb_format
from fastapi.testclient import TestClient
from vae.api import app
from vae.vae_utils import load_vae

from tests.image_loading_test_cases import load_images_for_test

client = TestClient(app)

examples_folder = Path(__file__).resolve().parent.joinpath("..", "data")
example_file = examples_folder.joinpath( "106818.jpg")

def tensor_to_stream(t: torch.Tensor) -> io.BytesIO:
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    return buf

class VAEApiTestCase(unittest.TestCase):

    def test_encoding_works(self):
        with open(example_file, "rb") as f:
            image_bytes = f.read()

        response = client.post(
            "/encode",
            files={"file": ("example.jpg", image_bytes, "image/jpeg")},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("application/octet-stream", response.headers["content-type"])

        import io
        buf = io.BytesIO(response.content)
        latent = torch.load(buf)

        self.assertEqual(7351, len(image_bytes))
        self.assertIsInstance(latent, torch.Tensor)
        self.assertEqual((4, 16, 16), latent.shape)
        self.assertEqual(4096, latent.numel() * latent.element_size(), "Approximate size in bytes should be a lot smaller")

    def test_decoding_works(self):
        vae = load_vae(model_file=None, device="cpu")
        test_image = load_images_for_test("000001.jpg").transpose(1, 2, 0)
        boxed_image = from_rgb_format(test_image, img_size=(128, 128))
        posterior = vae.encode(boxed_image.unsqueeze(0))
        z = posterior.latent_dist.mean

        response = client.post(
            "/decode",
            files={"file": ("latent.pt", tensor_to_stream(z.squeeze(0)), "application/octet-stream")},
        )
        img = Image.open(io.BytesIO(response.content))

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title("Decoded image")
        plt.show()
        plt.close()

        self.assertEqual(200, response.status_code)
        self.assertEqual("image/jpeg", response.headers["content-type"])


if __name__ == '__main__':
    unittest.main()
