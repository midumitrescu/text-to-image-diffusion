import io

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from data_loading.image_utils import to_rgb_format, from_rgb_format, ensure_correct_dimensions
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette import status
from starlette.responses import StreamingResponse
from transformers.image_transforms import to_pil_image
from vae.train import load_vae

app = FastAPI(title="VAE Downsampling API")


def encode(x):
    posterior = model.encode(x)
    return posterior.latent_dist.sample()

def decode(x):
    return model.decode(x).sample

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model = load_vae(device="cpu", model_file=None)
model.eval()

to_tensor = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])

to_pil = T.ToPILImage()

def tensor_to_stream(t: torch.Tensor) -> io.BytesIO:
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    return buf

def image_to_stream(image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf

@app.post("/encode")
async def downsample_image(file: UploadFile = File(...)):
    image = await extract_image_from_http_request(file)
    img = ensure_correct_dimensions(image)
    x = from_rgb_format(np.array(img), img_size=(128, 128)).unsqueeze(0).to(device)

    with torch.no_grad():
        posterior = model.encode(x)
        z = posterior.latent_dist.mean  # (B,C,H_latent,W_latent)

    if z.ndim == 4:
        latent = z.squeeze(0)
    else:
        latent = z

    return StreamingResponse(
        tensor_to_stream(latent),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=latent.pt"},
    )

@app.post("/decode", status_code=status.HTTP_200_OK)
async def decode_image(file: UploadFile = File(...)):
    # Load latent tensor from uploaded file
    buf = io.BytesIO(await file.read())
    latent = torch.load(buf, map_location=device)
    # Ensure shape is (B, C, H, W)
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)

    with torch.no_grad():
        decoded = model.decode(latent).sample

    decoded_to_rgb = to_rgb_format(decoded.squeeze(0))
    pil_image = to_pil_image(decoded_to_rgb)

    return StreamingResponse(
        image_to_stream(pil_image),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=image.jpg"},
    )


async def extract_image_from_http_request(file):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    return image
