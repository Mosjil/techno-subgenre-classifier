import argparse
import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import pandas as pd
import os

from src.dataset import TechnoDataset
from src.models.cnn_bigru import ParallelCNNBiGRU
from src.utils.plots import plot_training_metrics
from src.utils.utils import parse_labels
from src.train import train

from src.config import preprocess, train_config

# TODO : Fichier de config global

def main():


    # Parser
    parser = argparse.ArgumentParser(description="Train a multi-label subgenre classifier")
    parser.add_argument("--csv_path", type=str, default=preprocess.preprocessed_csv)
    parser.add_argument("--data_dir", type=str, default=preprocess.spectrogram_dir)
    parser.add_argument("--batch_size", type=int, default=train_config.batch_size)
    parser.add_argument("--epochs", type=int, default=train_config.epochs)
    parser.add_argument("--lr", type=float, default=train_config.learning_rate)
    parser.add_argument("--model", type=str, default=train_config.model)
    parser.add_argument("--num_workers", type=int, default=train_config.num_workers)
    parser.add_argument("--val_split", type=float, default=train_config.val_split)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Unique output dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{args.model}_bs{args.batch_size}_{timestamp}"
    output_dir = os.path.join(train_config.generic_output_dir, run_name)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # CSV load, and dataset creation
    df = pd.read_csv(args.csv_path)
    all_labels = sorted({label for labels in df["subgenres"] for label in parse_labels(labels)})
    num_classes = len(all_labels)

    print(f"Detected {num_classes} unique subgenres:", all_labels)

    dataset = TechnoDataset(csv_path=args.csv_path, class_list=all_labels)

    # for i in range(100):
    #     mel, label = dataset[i]
    #     print(f"Sample {i}")
    #     print("Raw label:", dataset.df.iloc[i]["subgenres"])
    #     print("Parsed:", parse_labels(dataset.df.iloc[i]["subgenres"]))
    #     print("Vector:", label.tolist())
    #     print("-----")

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
        checkpoint_dir=checkpoint_dir,
    )

    # Plot
    if args.plots:
        print("Plotting training metrics")
        log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        plot_training_metrics(log_path, save_dir=f"{output_dir}/plots")


if __name__ == "__main__":
    main()
