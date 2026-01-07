import csv
import json
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from .config import PROCESSED_DIR, REPORTS_DIR
from .data import Bbbc041Crops, build_index
from .metrics import compute_confusion_matrix, compute_metrics
from .model import build_model
from .transforms import build_transforms
from .utils import ensure_dir, get_device
from .viz import plot_confusion_matrix


def _load_label_names(label_mode: str) -> List[str]:
    label_map_path = PROCESSED_DIR / f"label_map_{label_mode}.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if label_mode == "binary":
        label_names = ["uninfected", "infected"]
    else:
        label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    return label_names


def evaluate_model(
    model_path: Path,
    label_mode: str = "binary",
    splits: Optional[List[str]] = None,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    device: Optional[str] = None,
    backbone: str = "resnet34",
) -> Path:
    device = get_device(device)
    splits = splits or ["val", "test"]

    index_path = build_index(label_mode=label_mode)
    label_names = _load_label_names(label_mode)
    num_classes = len(label_names)

    model = build_model(num_classes=num_classes, pretrained=False, backbone=backbone)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    ensure_dir(REPORTS_DIR)
    metrics_path = REPORTS_DIR / "metrics.csv"

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "accuracy", "macro_f1", "balanced_accuracy", "num_samples"])

        for split in splits:
            ds = Bbbc041Crops(
                index_path=index_path,
                split=split,
                transform=build_transforms(train=False),
                max_samples=max_samples,
            )
            pin_memory = device.type == "cuda"
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            y_true, y_pred = [], []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1).cpu().tolist()
                    y_pred.extend(preds)
                    y_true.extend(labels.tolist())

            metrics = compute_metrics(y_true, y_pred)
            writer.writerow(
                [
                    split,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    f"{metrics['balanced_accuracy']:.4f}",
                    len(y_true),
                ]
            )

            if split == "test":
                cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
                plot_confusion_matrix(cm, label_names, REPORTS_DIR / "confusion_matrix.png")

    return metrics_path
