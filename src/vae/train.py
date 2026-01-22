import argparse
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from data_loading.data_loaders import load_data
from diffusers.optimization import get_cosine_schedule_with_warmup
from loguru import logger
from tqdm import tqdm
from vae.vae_utils import evaluate_vae, Checkpoint, load_vae

training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vae_for_training(model_file: Path):
    return load_vae(model_file, device=training_device)

def train_vae(image_size, epochs, batch_size, lr, model_file=None, beta_slope=0.01, validation_ratio=0.2):

    vae = load_vae_for_training(model_file)
    train_loader, test_loader = load_data(image_size, batch_size, center_mean=True, train_ratio=(1-validation_ratio), text_labels=False, images_dir=Path(__file__).parent.resolve().joinpath("..", "..", "img_align_celeba"))

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-3)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=400,
        num_training_steps=(len(train_loader) * epochs)
    )
    accelerator = Accelerator(gradient_accumulation_steps=2)
    vae, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_loader, lr_scheduler
    )

    logger.info("Starting training on {}", vae.device)


    initial_metric = evaluate_vae(vae, test_loader, return_examples=2)
    model_saver = Checkpoint(init_metric=initial_metric, metric_to_use="mean_mse")
    losses = []

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        if torch.cuda.is_available():
            logger.debug("Current CUDA memory utilization: {}. Max allocation {} MB", torch.cuda.memory_summary(), torch.cuda.max_memory_allocated() / 1024**2)

        for batch_idx, (data, _) in enumerate(progress_bar):
            x = data.float()
            posterior = vae.encode(x)

            mu = posterior.latent_dist.mean
            logvar = posterior.latent_dist.logvar
            z = posterior.latent_dist.sample()

            x_hat = vae.decode(z).sample
            reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            beta = 1 / (1 + np.exp(- beta_slope * (epoch - 0.3*epochs)))
            loss = reconstruction_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(
                loss=loss.item(),
                recon=reconstruction_loss.item(),
                kl=kl_loss.item(),
            )
        res = evaluate_vae(vae, test_loader, return_examples=0)
        losses.append({k: res[k] for k in ["mean_mse", "mean_kl", "mean"]})
        model_saver.step(epoch=epoch, metric=res, model=vae)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', '-i', type=tuple, default=(128, 128), help='The input image size in the VAE')
    parser.add_argument('--batch_size', '-b', type=int, default=5, help='Training and validation batch size')
    parser.add_argument('--validation_ratio', '-val', type=float, default=5, help='How much of the dataset should we use as validation?')

    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='Training and validation batch size')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs')

    parser.add_argument('--save-dir', '-s', type=str, default='./save/', help='Directory to save checkpoints')

    args = parser.parse_args()

    train_vae(image_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, validation_ratio=args.validation_ratio)
