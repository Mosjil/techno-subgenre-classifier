import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel

class ASTForAudio(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        embed_dim = self.backbone.config.hidden_size
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x : (B, 1, n_mels, T)
        x = F.interpolate(x, size=(128, 1024), mode="bilinear", align_corners=False)
        x = x.squeeze(1)  # (B, 128, 1024)

        outputs = self.backbone(input_values=x)  # ASTModel -> BaseModelOutputWithPooling [web:12]
        last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden_size)
        cls = last_hidden[:, 0]  # token [CLS]
        logits = self.head(cls)
        return logits