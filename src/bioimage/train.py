import csv
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .data import Bbbc041Crops, build_index
from .metrics import compute_metrics
from .model import build_model
from .transforms import build_transforms
from .utils import ensure_dir, get_device, save_json, set_seed


def _class_weights(labels, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.sum() * num_classes
    return weights


def train_model(
    label_mode: str = "binary",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.1,
    dropout: float = 0.4,
    backbone: str = "resnet34",
    freeze_backbone: bool = False,
    unfreeze_layer4: bool = False,
    layer4_lr_mult: float = 0.01,
    seed: int = 42,
    max_samples: Optional[int] = None,
    balance: bool = True,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Path:
    set_seed(seed)
    device = get_device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"Using CUDA: {name} (capability {cap[0]}.{cap[1]})")
    else:
        print("Using CPU")

    index_path = build_index(label_mode=label_mode, seed=seed)

    train_ds = Bbbc041Crops(
        index_path=index_path,
        split="train",
        transform=build_transforms(train=True),
        max_samples=max_samples,
        seed=seed,
    )
    val_ds = Bbbc041Crops(
        index_path=index_path,
        split="val",
        transform=build_transforms(train=False),
        max_samples=max_samples,
        seed=seed,
    )

    label_map_path = PROCESSED_DIR / f"label_map_{label_mode}.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    model = build_model(
        num_classes=num_classes,
        pretrained=True,
        dropout=dropout,
        backbone=backbone,
    ).to(device)
    if freeze_backbone and unfreeze_layer4:
        raise ValueError("Only one of freeze_backbone or unfreeze_layer4 can be set.")
    if freeze_backbone or unfreeze_layer4:
        for param in model.backbone.parameters():
            param.requires_grad = False
    if unfreeze_layer4:
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True

    train_labels = [row.label_id for row in train_ds.rows]
    class_weights = _class_weights(train_labels, num_classes).to(device)
    loss_weights = None
    sampler = None
    if balance:
        sample_weights = class_weights.cpu().numpy()[train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=label_smoothing,
    )

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    if freeze_backbone:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif unfreeze_layer4:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.layer4.parameters(), "lr": lr * layer4_lr_mult},
                {"params": model.classifier.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": lr * 0.1},
                {"params": model.classifier.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )

    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    best_path = MODELS_DIR / f"bbbc041_{label_mode}_best.pt"
    best_metric = -1.0

    log_path = REPORTS_DIR / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_balanced_accuracy"])

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            logged_cuda = False
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if device.type == "cuda" and not logged_cuda:
                    torch.cuda.synchronize()
                    mem = torch.cuda.max_memory_allocated() / (1024**2)
                    print(f"CUDA memory allocated after first batch: {mem:.1f} MB")
                    logged_cuda = True
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = torch.argmax(logits, dim=1)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())
            val_loss /= len(val_ds)
            metrics = compute_metrics(y_true, y_pred)

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    f"{metrics['balanced_accuracy']:.4f}",
                ]
            )
            f.flush()

            if metrics["macro_f1"] > best_metric:
                best_metric = metrics["macro_f1"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)

    save_json(
        {
            "label_mode": label_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "dropout": dropout,
            "seed": seed,
            "max_samples": max_samples,
            "balance": balance,
            "num_classes": num_classes,
            "backbone": backbone,
            "freeze_backbone": freeze_backbone,
            "unfreeze_layer4": unfreeze_layer4,
            "layer4_lr_mult": layer4_lr_mult,
        },
        REPORTS_DIR / "train_config.json",
    )

    return best_path
