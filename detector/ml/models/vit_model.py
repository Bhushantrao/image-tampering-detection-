import torch
import torch.nn as nn
import timm


class ViTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=2
        )

        # Dropout improves generalization
        self.model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                self.model.head.in_features,
                2
            )
        )

    def forward(self, x):

        return self.model(x)