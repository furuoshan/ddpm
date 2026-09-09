from forward_noising import forward_diffusion_sample
from unet import SimpleUnet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging
import os
import torchvision
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = "./trained_models"
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ddpm_latest.pth")
SAVE_EVERY = 10


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logging.info(f"Saved checkpoint at epoch {epoch} to {path}")


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("epoch", 0))
    model.load_state_dict(checkpoint)
    return 0


if __name__ == "__main__":
    model = SimpleUnet()
    T = 300
    BATCH_SIZE = 256
    epochs = 10

    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    if os.path.isfile(LATEST_CHECKPOINT):
        start_epoch = load_checkpoint(model, optimizer, LATEST_CHECKPOINT, device)
        logging.info(f"Resumed from {LATEST_CHECKPOINT} at epoch {start_epoch}")
    else:
        logging.info("No checkpoint found, training from scratch")

    end_epoch = start_epoch + epochs
    logging.info(f"Training epochs {start_epoch} -> {end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        for batch_idx, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t, device=device)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {loss.item()}")

        completed_epoch = epoch + 1
        if completed_epoch % SAVE_EVERY == 0:
            snapshot_path = os.path.join(CHECKPOINT_DIR, f"ddpm_epoch_{completed_epoch}.pth")
            save_checkpoint(model, optimizer, completed_epoch, snapshot_path)
            save_checkpoint(model, optimizer, completed_epoch, LATEST_CHECKPOINT)

    if end_epoch % SAVE_EVERY != 0:
        save_checkpoint(model, optimizer, end_epoch, LATEST_CHECKPOINT)
