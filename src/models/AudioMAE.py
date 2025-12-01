import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AudioMAEForAudio(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = timm.create_model("hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True,
            num_classes=0,
        )

        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim == 3:
            # (B, n_mels, T) -> (B, 1, n_mels, T)
            x = x.unsqueeze(1)

        x = F.interpolate(x, size=(1024, 128), mode="bilinear", align_corners=False)

        feats = self.backbone(x)
        logits = self.head(feats)

        return logits