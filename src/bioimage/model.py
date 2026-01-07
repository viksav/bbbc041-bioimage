import torch
from torch import nn
from torchvision import models


_RESNET_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
}


def _build_resnet_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name not in _RESNET_BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Options: {', '.join(sorted(_RESNET_BACKBONES))}")
    ctor, weight_enum = _RESNET_BACKBONES[name]
    weights = weight_enum if pretrained else None
    backbone = ctor(weights=weights)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, num_features


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.4,
        backbone: str = "resnet34",
    ) -> None:
        super().__init__()
        backbone_model, num_features = _build_resnet_backbone(backbone, pretrained=pretrained)
        self.backbone = backbone_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_model(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.4,
    backbone: str = "resnet34",
) -> ResNetClassifier:
    return ResNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        backbone=backbone,
    )
