import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Calcule les métriques principales pour un problème multi-label.
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Sigmoid
    y_prob = 1 / (1 + np.exp(-y_pred))

    # Binaire
    y_hat = (y_prob >= threshold).astype(int)

    # Metrics
    f1_macro = f1_score(y_true, y_hat, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_hat, average="micro", zero_division=0)
    precision = precision_score(y_true, y_hat, average="macro", zero_division=0)
    recall = recall_score(y_true, y_hat, average="macro", zero_division=0)

    metrics = {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
    }
    return metrics


# Find best treshold