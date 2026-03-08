import os
import time
import json
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import FastCNN
from plotting import *


@dataclass
class Config:
    name: str
    conv_filters: Tuple[int, int]
    fc1_size: int
    dropout_p: float
    pool_kernel: int
    pool_stride: int


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_eval(cfg: Config, device, epochs=10, batch_size=128, val_split=0.1, subset=None, dataset_name: Optional[str] = None):
    # enforce minimum number of epochs before early stopping / LR scheduling
    min_epochs = 50
    if epochs < min_epochs:
        epochs = min_epochs
    # image transforms: resize to 224x224, ensure 3 channels, normalize with ImageNet stats
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # prefer explicit dataset layout under dataset/<dataset_name>/train and dataset/<dataset_name>/test (ImageFolder)
    if dataset_name:
        train_dir = os.path.join("dataset", dataset_name, "train")
        test_dir = os.path.join("dataset", dataset_name, "test")
    else:
        train_dir = os.path.join("dataset", "train")
        test_dir = os.path.join("dataset", "test")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=transform)
        val_ds = datasets.ImageFolder(test_dir, transform=transform)

        if subset is not None and subset < len(train_ds):
            train_ds, _ = random_split(train_ds, [subset, len(train_ds) - subset])
    else:
        # fallback to MNIST (torchvision) if dataset folders are not present
        ds_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

        if subset is not None and subset < len(ds_full):
            ds_full, _ = random_split(ds_full, [subset, len(ds_full) - subset])

        n_val = max(1, int(len(ds_full) * val_split))
        n_train = len(ds_full) - n_val
        train_ds, val_ds = random_split(ds_full, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # determine number of classes from dataset and use 3-channel input
    def _infer_num_classes(ds):
        if hasattr(ds, "classes"):
            return len(ds.classes)
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "classes"):
            return len(ds.dataset.classes)
        return 10

    num_classes = _infer_num_classes(train_ds)
    model = FastCNN(in_channels=3, conv_filters=cfg.conv_filters, fc1_size=cfg.fc1_size, dropout_p=cfg.dropout_p, pool_kernel=cfg.pool_kernel, pool_stride=cfg.pool_stride, num_classes=num_classes).to(device)

    # print GAP input shape once (channels and spatial size) before training starts
    try:
        dummy = torch.randn(1, 3, 224, 224, device=device)
        gap_shape = model.gap_input_shape(dummy)
        print(f"{cfg.name} GAP input shape (batch,channels,H,W): {gap_shape}")
    except Exception as e:
        print("Could not infer GAP input shape:", e)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    # LR scheduler: reduce LR by factor 0.5 if val_loss doesn't improve for 4 epochs, min lr = 1e-6
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # early stopping params
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 7

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            b = x.size(0)
            running_loss += loss.item() * b
            running_acc += (torch.argmax(out, dim=1) == y).sum().item()
            seen += b

        train_loss = running_loss / seen
        train_acc = running_acc / seen

        model.eval()
        v_loss = 0.0
        v_acc_cnt = 0
        v_seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                b = x.size(0)
                v_loss += loss.item() * b
                v_acc_cnt += (torch.argmax(out, dim=1) == y).sum().item()
                v_seen += b

        val_loss = v_loss / v_seen
        val_acc = v_acc_cnt / v_seen

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # early stopping check (update counters)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # step LR scheduler based on validation loss only after min_epochs
        try:
            if epoch >= min_epochs:
                scheduler.step(val_loss)
        except Exception:
            pass

        # report current learning rate (first param group)
        lr = opt.param_groups[0]["lr"]
        print(f"{cfg.name} E{epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} " f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.1e}")

        # early stopping only allowed after min_epochs have completed
        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in validation loss for {patience} epochs. Stopping training.")
            break

    return model, history


def main(dataset_name: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # if dataset_name parameter passed, use it; otherwise auto-detect
    if dataset_name is None:
        if os.path.isdir(os.path.join("dataset", "MNIST")):
            dataset_name = "MNIST"
        elif os.path.isdir(os.path.join("dataset", "train")):
            dataset_name = "BT"
        else:
            dataset_name = "MNIST"

    configs: List[Config] = [
        Config("Model-01", (8, 16), 50, 0.3, 2, 2),
        Config("Model-02", (16, 32), 100, 0.5, 2, 2),
        Config("Model-03", (32, 64), 150, 0.5, 2, 2),
        Config("Model-04", (16, 32), 50, 0.2, 2, 1),
        Config("Model-05", (8, 32), 200, 0.6, 3, 2),
        Config("Model-06", (12, 24), 80, 0.4, 2, 2),
        Config("Model-07", (20, 40), 120, 0.5, 2, 2),
        Config("Model-08", (16, 48), 64, 0.25, 2, 2),
        Config("Model-09", (10, 20), 32, 0.1, 2, 2),
        Config("Model-10", (24, 48), 256, 0.7, 3, 3),
    ]

    out_dir = os.path.join("results", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "histories"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    results = []

    # default: full MNIST (60000) split with 10% val. To speed up testing, pass subset env var or edit below.
    subset = None  # set to int like 5000 to debug quickly
    epochs = 200
    batch_size = 32

    for cfg in configs:
        t0 = time.time()
        model, history = train_eval(cfg, device=device, epochs=epochs, batch_size=batch_size, val_split=0.1, subset=subset, dataset_name=dataset_name)
        dt = time.time() - t0

        # save model state and history
        # model_path = os.path.join(out_dir, "models", f"{cfg.name}.pt")
        # torch.save(model.state_dict(), model_path)

        hist_path = os.path.join(out_dir, "histories", f"{cfg.name}_history.json")
        with open(hist_path, "w") as f:
            json.dump({"config": asdict(cfg), "history": history}, f)

        final_val_loss = history["val_loss"][-1]
        final_val_acc = history["val_acc"][-1]
        results.append({"name": cfg.name, "val_loss": final_val_loss, "val_acc": final_val_acc, "time_s": dt, **asdict(cfg)})

    # sort by val_loss
    results_sorted = sorted(results, key=lambda r: r["val_loss"])
    csv_path = os.path.join(out_dir, "results_ranked.json")
    with open(csv_path, "w") as f:
        json.dump(results_sorted, f, indent=2)

    print("\nRanking (best -> worst by val_loss):")
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['name']} val_loss={r['val_loss']:.4f} val_acc={r['val_acc']:.4f} time={r['time_s']:.1f}s")

    # call plotting with the correct results folder
    from plotting import plot_all

    plot_all(hist_dir=os.path.join(out_dir, "histories"), out_dir=os.path.join(out_dir, "plots"))


if __name__ == "__main__":
    main()
