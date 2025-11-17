import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel

class ViTSmallMAEForAudio(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ViTMAEModel.from_pretrained(
            "facebook/vit-mae-small",
            add_pooling_layer=True
        )

        embed_dim = 384

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Dupliquer les spectrogrammes en 3 canaux car .npy en 2D
        x = x.repeat(1, 3, 1, 1)

        # Passer dans le ViTMAE
        out = self.backbone(pixel_values=x)

        cls = out.pooler_output

        return self.head(cls)
