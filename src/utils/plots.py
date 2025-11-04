import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# TODO : Modifier le path si jamais passage en dynamique.
# TODO : Modifier les noms des graphs dynamiquement.

def plot_training_metrics(log_path="outputs/checkpoints/training_log.csv", save_dir="outputs/plots"):

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} epochs from {log_path}")

    os.makedirs(save_dir, exist_ok=True)

    # Metrics list
    metrics = [col for col in df.columns if col not in ["epoch"]]

    # Loss curves
    if {"train_loss", "val_loss"} <= set(df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="red")
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(save_dir, "plot_loss_curves.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.show()

    # Other metrics
    metric_cols = [m for m in metrics if m not in ["train_loss", "val_loss"]]
    for metric in metric_cols:
        plt.figure(figsize=(6, 4))
        plt.plot(df["epoch"], df[metric], label=metric.capitalize(), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} per Epoch")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"plot_{metric}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.show()

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/training_log.csv"
    plot_training_metrics(log_path)