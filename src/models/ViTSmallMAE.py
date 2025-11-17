import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel

class ViTSmallMAEForAudio(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Charger le MAE Small pré-entraîné
        self.backbone = ViTMAEModel.from_pretrained(
            "facebook/vit-mae-small",
            add_pooling_layer=True
        )

        # Dimension de l'embedding CLS
        embed_dim = 384

        # Head de classification
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 1, H, W) car ton .npy est 2D → on ajoute juste un canal
        """

        x = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Dupliquer les spectrogrammes en 3 canaux
        # (B, 1, H, W) -> (B, 3, H, W)
        x = x.repeat(1, 3, 1, 1)

        # Passer dans le ViT-MAE
        out = self.backbone(pixel_values=x)

        # Récupérer le CLS (déjà poolé)
        cls = out.pooler_output  # shape (B, 384)

        # Classification
        return self.head(cls)
