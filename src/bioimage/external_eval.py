import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .config import REPORTS_DIR
from .external import prepare_nih_malaria
from .metrics import compute_confusion_matrix, compute_metrics
from .model import build_model
from .transforms import build_transforms
from .utils import ensure_dir, get_device
from .viz import plot_confusion_matrix


class RemappedImageFolder(Dataset):
    def __init__(self, root: Path, transform, label_map: Dict[str, int]) -> None:
        self.base = ImageFolder(root=str(root), transform=transform)
        self.label_map = {k.lower(): v for k, v in label_map.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        label_name = self.base.classes[label]
        mapped = self.label_map.get(label_name.lower())
        if mapped is None:
            raise ValueError(f"Unknown label {label_name}. Available: {self.base.classes}")
        return image, mapped


def evaluate_external_nih(
    model_path: Path,
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[str] = None,
    backbone: str = "resnet34",
) -> Path:
    device = get_device(device)
    pin_memory = device.type == "cuda"

    data_root = prepare_nih_malaria()
    label_map = {"uninfected": 0, "parasitized": 1}
    label_names = ["uninfected", "infected"]

    dataset = RemappedImageFolder(
        root=data_root,
        transform=build_transforms(train=False),
        label_map=label_map,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(num_classes=2, pretrained=False, backbone=backbone)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    metrics = compute_metrics(y_true, y_pred)

    ensure_dir(REPORTS_DIR)
    metrics_path = REPORTS_DIR / "external_nih_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "accuracy", "macro_f1", "balanced_accuracy", "num_samples"])
        writer.writerow([
            "NIH_Malaria_Cell_Images",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['macro_f1']:.4f}",
            f"{metrics['balanced_accuracy']:.4f}",
            len(y_true),
        ])

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)
    plot_confusion_matrix(cm, label_names, REPORTS_DIR / "external_nih_confusion_matrix.png")

    return metrics_path
