import argparse
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os

from dataset import TechnoDataset
from models.cnn_bigru import ParallelCNNBiGRU
from utils.plots import plot_training_metrics
from utils.utils import parse_labels
from train import train

# TODO : Fichier de config global

def main():

    # Parser
    parser = argparse.ArgumentParser(description="Train a multi-label subgenre classifier")
    parser.add_argument("--csv_path", type=str, default="data/processed.csv")
    parser.add_argument("--data_dir", type=str, default="data/specs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="cnn_bigru")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # CSV load, and dataset creation
    df = pd.read_csv(args.csv_path)
    all_labels = sorted({label for labels in df["subgenres"] for label in parse_labels(labels)})
    num_classes = len(all_labels)

    print(f"Detected {num_classes} unique subgenres:", all_labels)

    dataset = TechnoDataset(csv_path=args.csv_path, class_list=all_labels)

    # Split train/val
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Ini model
    if args.model == "cnn_bigru":
        model = ParallelCNNBiGRU(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Training
    train(
        train_loader=train_dl,
        val_loader=val_dl,
        model=model,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Plot
    if args.plots:
        print("Plotting training metrics")
        log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        plot_training_metrics(log_path, save_dir="outputs/plots")


if __name__ == "__main__":
    main()
