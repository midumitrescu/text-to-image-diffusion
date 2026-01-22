import io
from datetime import datetime
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from accelerate.commands.merge import description
from pydantic import BaseModel, Field

from data_loading.image_utils import to_rgb_format, from_rgb_format, ensure_correct_dimensions
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette import status
from starlette.responses import StreamingResponse
from transformers.image_transforms import to_pil_image
from vae.train import load_vae

import uvicorn

service_startup_time = datetime.now()
app = FastAPI(title="VAE Downsampling API",
              description="A subcomponent of Text to Image Difusion, used for downsampling to a uniform features space",
              version="1.0.0",
              docs_url="/docs")


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

class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["running", "model_not_loaded"] = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Model loaded status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")

@app.get("/health",  tags=["monitoring"])
async def health_check():
    """
    Health check endpoint for monitoring service status.

    Returns service health, model status, and uptime information.
    """
    uptime = (datetime.now() - service_startup_time).total_seconds()

    return (
        HealthResponse(
        status="running" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat(),
    ))

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)