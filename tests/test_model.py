import torch

from bioimage.model import build_model


def test_model_forward() -> None:
    model = build_model(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 3)
