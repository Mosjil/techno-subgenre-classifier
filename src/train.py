import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
from src.utils.metrics import compute_metrics
from src.utils.early_stopping import EarlyStopping
from src.utils.lr_scheduler import LRSchedulerWrapper


def train(train_loader, val_loader, model, num_epochs=30, lr=1e-3, device=None, checkpoint_dir="outputs/checkpoints"):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping()
    scheduler = LRSchedulerWrapper(optimizer, mode='max') # Maximizing F1

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_f1 = 0.0

    # Log CSV
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "f1_macro", "f1_micro", "precision", "recall"])

    # Main loop
    for epoch in range(1, num_epochs + 1):

        # Train
        model.train()
        train_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()

            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_true, all_pred = [], []

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]"):
                xb, yb = xb.to(device), yb.to(device).float()
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

                all_true.append(yb.cpu())
                all_pred.append(preds.cpu())

        val_loss /= len(val_loader)
        y_true = torch.cat(all_true, dim=0)
        y_pred = torch.cat(all_pred, dim=0)

        metrics = compute_metrics(y_true, y_pred)
        f1_macro = metrics["f1_macro"]

        scheduler.step(f1_macro)
        current_lr = scheduler.get_lr()

        print(
            f"\nEpoch {epoch}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"F1 macro: {f1_macro:.4f} | "
            f"LR: {current_lr:.4f}"
        )

        # Values log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                metrics["f1_macro"],
                metrics["f1_micro"],
                metrics["precision"],
                metrics["recall"]
            ])

        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"New best model saved ({path})\n")

        # Early stopping
        early_stopper(f1_macro)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}, f1 macro: {f1_macro:.4f}")
            break

    print(f"Finished. Best F1 macro: {best_f1:.4f}")
