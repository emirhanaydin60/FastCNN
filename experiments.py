import csv
import time
from dataclasses import dataclass, asdict
from typing import Tuple, List

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import FastCNN


@dataclass
class Config:
    name: str
    conv_filters: Tuple[int, int]
    fc1_size: int
    dropout_p: float
    pool_kernel: int
    pool_stride: int


def train_once(cfg: Config, epochs=3, batch_size=128, subset=None, device=None) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    if subset is not None and subset < len(ds):
        ds = Subset(ds, list(range(subset)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = FastCNN(conv_filters=cfg.conv_filters, fc1_size=cfg.fc1_size, dropout_p=cfg.dropout_p, pool_kernel=cfg.pool_kernel, pool_stride=cfg.pool_stride).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    last_loss = None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        last_loss = running / len(loader.dataset)
    return last_loss


def main():
    # 10 different configs (vary filters, fc neurons, dropout, pool stride)
    configs: List[Config] = [
        Config("cfg1", (8, 16), 50, 0.3, 2, 2),
        Config("cfg2", (16, 32), 100, 0.5, 2, 2),
        Config("cfg3", (32, 64), 150, 0.5, 2, 2),
        Config("cfg4", (16, 32), 50, 0.2, 2, 1),
        Config("cfg5", (8, 32), 200, 0.6, 3, 2),
        Config("cfg6", (12, 24), 80, 0.4, 2, 2),
        Config("cfg7", (20, 40), 120, 0.5, 2, 2),
        Config("cfg8", (16, 48), 64, 0.25, 2, 2),
        Config("cfg9", (10, 20), 32, 0.1, 2, 2),
        Config("cfg10", (24, 48), 256, 0.7, 3, 3),
    ]

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running experiments on", device)

    # to keep runtime modest for demo, use subset; change subset=None to run full dataset
    subset = 5000
    epochs = 3
    batch_size = 128

    for cfg in configs:
        t0 = time.time()
        loss = train_once(cfg, epochs=epochs, batch_size=batch_size, subset=subset, device=device)
        dt = time.time() - t0
        print(f"{cfg.name}: loss={loss:.4f} time={dt:.1f}s cfg={asdict(cfg)}")
        results.append({**asdict(cfg), "loss": loss, "time_s": dt})

    # sort by loss ascending
    results_sorted = sorted(results, key=lambda r: r["loss"])

    # save CSV
    keys = list(results_sorted[0].keys())
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_sorted)

    print("\nRanking (best -> worst):")
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['name']} loss={r['loss']:.4f} time={r['time_s']:.1f}s")


if __name__ == "__main__":
    main()
