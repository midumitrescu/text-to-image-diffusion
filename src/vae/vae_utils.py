from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from loguru import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Checkpoint:
    def __init__(self, folder="models/", mode="min", init_metric: dict = None, metric_to_use = "mean_mse", experiment_label=None):
        self.experiment_label = "" if experiment_label is None else f"_{experiment_label}"

        self.metric_to_use = metric_to_use
        if init_metric:
            self.best_loss = init_metric[self.metric_to_use]
        else:
            self.best_loss = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = 0
        self.mode = mode
        self.folder = folder
        Path(self.folder).mkdir(parents=True, exist_ok=True)

    def step(self, epoch: int, beta: float, metric=dict, model: torch.nn.Module = None):

        loss = metric[self.metric_to_use]
        improved = loss < self.best_loss if self.mode == "min" else loss > self.best_loss
        logger.debug("{}: Best MSE = {}, for epoch {}. Current MSE = {} at epoch {}. Improvement? {}", self.experiment_label, self.best_loss, self.best_epoch, loss, epoch, improved)
        if improved:
            self.best_loss = loss
        torch.save({
            "model": model.state_dict(),
            "metric": metric,
            "epoch": epoch,
            "beta": f"{beta: .6f}",
        }, f"{self.folder}/vae{self.experiment_label}_{epoch}.pt")


def evaluate_vae(vae, val_loader, device=device, return_examples=4):
    """
    Evaluate a VAE on a validation set, reporting per-image MSE and KL divergence.

    Args:
        vae: the AutoencoderKL model (or any VAE)
        val_loader: PyTorch DataLoader for the validation dataset
        device: "cuda" or "cpu"
        return_examples: number of examples to return for visualization

    Returns:
        dict with:
            - mean_mse: mean (per image) MSE on validation set
            - mean_latent_layer_kullback_leibler_divergence: mean KL divergence in the latent layer
            - examples: tuple (x, x_hat) of a few original and reconstructed images
    """
    vae.eval()
    total_mse = 0.0
    batch_kl = 0.0
    total_samples = 0

    example_x = []
    example_x_hat = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device).float()
            batch_size, _, _, _ = x.shape
            total_samples += batch_size
            # Encode
            posterior = vae.encode(x)
            mu = posterior.latent_dist.mean
            logvar = posterior.latent_dist.logvar
            z = posterior.latent_dist.mean
            # Decode
            x_hat = vae.decode(z).sample

            # Reconstruction loss: sum over pixels per image
            pixel_mse = F.mse_loss(x_hat, x, reduction="none")  # [B, C, H, W]
            image_mse = pixel_mse.view(batch_size, -1).sum(dim=1)  # sum over pixels
            total_mse += image_mse.sum().item()  # sum over batch

            # KL divergence per image
            image_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
            batch_kl += image_kl.sum().item()

            # Save first few examples for visualization
            if len(example_x) < return_examples:
                n = min(return_examples - len(example_x), batch_size)
                example_x.append(x[:n].cpu())
                example_x_hat.append(x_hat[:n].cpu())

    # Compute averages per image
    mean_mse = total_mse / total_samples
    mean_kl = batch_kl / total_samples

    # Concatenate example images
    if example_x:
        example_x = torch.cat(example_x, dim=0)
        example_x_hat = torch.cat(example_x_hat, dim=0)

    results = {
        "mean_mse": mean_mse,
        "mean_kl": mean_kl,
        "mean": mean_mse + mean_kl,
        "examples": (example_x, example_x_hat) if example_x is not None else None
    }

    return results

def load_vae(model_file: None, device="cpu"):
    if model_file is None:
        return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    return {}