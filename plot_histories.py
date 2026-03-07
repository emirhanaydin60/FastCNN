import os
import json
from glob import glob
import matplotlib.pyplot as plt


def load_histories(hist_dir="results/histories"):
    paths = sorted(glob(os.path.join(hist_dir, "*_history.json")))
    histories = {}
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            # data may be {"config":..., "history":{...}}
            if "history" in data:
                name = os.path.splitext(os.path.basename(p))[0].replace("_history", "")
                histories[name] = data["history"]
            else:
                name = os.path.splitext(os.path.basename(p))[0]
                histories[name] = data
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    return histories


def plot_history(name, history, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="train_loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="val_loss", marker="o")
    axes[0].set_title(f"{name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="train_acc", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="val_acc", marker="o")
    axes[1].set_title(f"{name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_all(hist_dir="results/histories", out_dir="results/plots", combined=True):
    histories = load_histories(hist_dir)
    if not histories:
        print("No histories found in", hist_dir)
        return

    # per-model plots
    for name, hist in histories.items():
        # validate keys
        if not all(k in hist for k in ("train_loss", "val_loss", "train_acc", "val_acc")):
            print(f"Skipping {name}, missing keys in history")
            continue
        plot_history(name, hist, out_dir=out_dir)

    # combined plots (val_loss and val_acc across models)
    if combined:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for name, hist in histories.items():
            if "val_loss" in hist:
                plt.plot(range(1, len(hist["val_loss"]) + 1), hist["val_loss"], label=name)
        plt.title("Validation Loss - All Models")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.legend()
        plt.grid(True)
        pth = os.path.join(out_dir, "all_val_loss.png")
        plt.savefig(pth)
        plt.close()
        print(f"Saved combined val_loss plot: {pth}")

        plt.figure(figsize=(8, 6))
        for name, hist in histories.items():
            if "val_acc" in hist:
                plt.plot(range(1, len(hist["val_acc"]) + 1), hist["val_acc"], label=name)
        plt.title("Validation Accuracy - All Models")
        plt.xlabel("Epoch")
        plt.ylabel("Val Accuracy")
        plt.legend()
        plt.grid(True)
        pth = os.path.join(out_dir, "all_val_acc.png")
        plt.savefig(pth)
        plt.close()
        print(f"Saved combined val_acc plot: {pth}")


if __name__ == "__main__":
    plot_all()
