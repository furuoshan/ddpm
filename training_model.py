from forward_noising import forward_diffusion_sample
from unet import SimpleUnet
from dataloader import load_transformed_dataset
from ema import EMA
import torch.nn.functional as F
import torch
from torch.optim import Adam
import json
import logging
import os
import torchvision
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = "./trained_models"
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ddpm_latest.pth")
LOSS_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "loss_history.json")
SAVE_EVERY = 1
EMA_DECAY = 0.9999
EPOCH_CHECKPOINT_PREFIX = "ddpm_epoch_"


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)


def save_loss_history(loss_history):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp_path = f"{LOSS_HISTORY_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)
    os.replace(tmp_path, LOSS_HISTORY_PATH)


def save_checkpoint(model, optimizer, ema, epoch, path, loss=None, loss_history=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "loss": None if loss is None else float(loss),
        "loss_history": list(loss_history or []),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "ema_decay": ema.decay,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    extra = f" loss={float(loss):.6f}" if loss is not None else ""
    logging.info(f"Saved checkpoint at epoch {epoch} to {path}{extra}")


def find_resume_checkpoint():
    if os.path.isfile(LATEST_CHECKPOINT):
        return LATEST_CHECKPOINT
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    best_path = None
    best_epoch = -1
    prefix = EPOCH_CHECKPOINT_PREFIX
    suffix = ".pth"
    for name in os.listdir(CHECKPOINT_DIR):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        epoch_text = name[len(prefix) : -len(suffix)]
        if not epoch_text.isdigit():
            continue
        epoch = int(epoch_text)
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = os.path.join(CHECKPOINT_DIR, name)
    return best_path


def load_loss_history(checkpoint):
    history = []
    if isinstance(checkpoint, dict):
        raw = checkpoint.get("loss_history") or []
        if isinstance(raw, list):
            history = raw
    if os.path.isfile(LOSS_HISTORY_PATH):
        try:
            with open(LOSS_HISTORY_PATH, encoding="utf-8") as f:
                file_history = json.load(f)
            if isinstance(file_history, list) and len(file_history) > len(history):
                history = file_history
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning(f"Could not read {LOSS_HISTORY_PATH}: {exc}")
    return history


def load_checkpoint(model, optimizer, ema, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = {"epoch": 0, "loss": None, "loss_history": []}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema is not None:
            if "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
                if "ema_decay" in checkpoint:
                    ema.decay = checkpoint["ema_decay"]
            else:
                ema.load_state_dict(model.state_dict())
                logging.info("Checkpoint has no EMA weights; initialized EMA from model")
        loss = checkpoint.get("loss")
        meta["epoch"] = int(checkpoint.get("epoch", 0))
        meta["loss"] = None if loss is None else float(loss)
        meta["loss_history"] = load_loss_history(checkpoint)
        return meta
    model.load_state_dict(checkpoint)
    if ema is not None:
        ema.load_state_dict(model.state_dict())
    return meta


if __name__ == "__main__":
    model = SimpleUnet()
    T = 300
    BATCH_SIZE = 256

    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    ema = EMA(model, decay=EMA_DECAY)
    logging.info(f"EMA decay: {EMA_DECAY}")

    start_epoch = 0
    last_loss = None
    loss_history = []
    resume_path = find_resume_checkpoint()
    if resume_path is not None:
        meta = load_checkpoint(model, optimizer, ema, resume_path, device)
        start_epoch = meta["epoch"]
        last_loss = meta["loss"]
        loss_history = meta["loss_history"]
        if last_loss is None:
            logging.info(f"Resumed from {resume_path} at epoch {start_epoch} (no loss in checkpoint)")
        else:
            logging.info(f"Resumed from {resume_path} at epoch {start_epoch} last loss={last_loss:.6f}")
    else:
        logging.info("No checkpoint found, training from scratch")

    logging.info(f"Training from epoch {start_epoch} until interrupted (Ctrl+C)")

    epoch = start_epoch
    epoch_loss_sum = 0.0
    epoch_batches = 0
    try:
        while True:
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for batch_idx, (batch, _) in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, batch, t, device=device)
                loss.backward()
                optimizer.step()
                ema.update(model)

                last_loss = loss.item()
                epoch_loss_sum += last_loss
                epoch_batches += 1

                if batch_idx % 50 == 0:
                    logging.info(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {last_loss}")

            completed_epoch = epoch + 1
            mean_loss = epoch_loss_sum / max(epoch_batches, 1)
            loss_history.append({"epoch": completed_epoch, "loss": mean_loss})
            logging.info(f"Epoch {completed_epoch} mean loss: {mean_loss:.6f}")
            if completed_epoch % SAVE_EVERY == 0:
                snapshot_path = os.path.join(CHECKPOINT_DIR, f"{EPOCH_CHECKPOINT_PREFIX}{completed_epoch}.pth")
                save_checkpoint(
                    model, optimizer, ema, completed_epoch, snapshot_path,
                    loss=mean_loss, loss_history=loss_history,
                )
                save_checkpoint(
                    model, optimizer, ema, completed_epoch, LATEST_CHECKPOINT,
                    loss=mean_loss, loss_history=loss_history,
                )
                save_loss_history(loss_history)
            epoch += 1
    except KeyboardInterrupt:
        interrupt_epoch = epoch + 1
        if epoch_batches > 0:
            mean_loss = epoch_loss_sum / epoch_batches
            loss_history.append({"epoch": interrupt_epoch, "loss": mean_loss})
        else:
            mean_loss = last_loss
        logging.info(
            f"Interrupted during epoch {epoch}, saving weights as epoch {interrupt_epoch}"
            + (f" loss={mean_loss:.6f}" if mean_loss is not None else "")
        )
        snapshot_path = os.path.join(CHECKPOINT_DIR, f"{EPOCH_CHECKPOINT_PREFIX}{interrupt_epoch}.pth")
        save_checkpoint(
            model, optimizer, ema, interrupt_epoch, snapshot_path,
            loss=mean_loss, loss_history=loss_history,
        )
        save_checkpoint(
            model, optimizer, ema, interrupt_epoch, LATEST_CHECKPOINT,
            loss=mean_loss, loss_history=loss_history,
        )
        save_loss_history(loss_history)
