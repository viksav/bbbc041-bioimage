import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .config import PROCESSED_DIR, REPORTS_DIR
from .data import Bbbc041Crops, build_index
from .model import build_model
from .transforms import build_transforms
from .utils import ensure_dir, get_device
from .viz import GradCAM, plot_umap, save_gradcam_grid


def _load_label_names(label_mode: str) -> List[str]:
    label_map_path = PROCESSED_DIR / f"label_map_{label_mode}.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if label_mode == "binary":
        return ["uninfected", "infected"]
    return [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]


def _collect_embeddings(
    model,
    loader,
    device,
    max_samples: int,
) -> Tuple[np.ndarray, List[int]]:
    embeddings = []
    labels = []
    count = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            logits, feats = model(images, return_features=True)
            embeddings.append(feats.cpu().numpy())
            labels.extend(batch_labels.tolist())
            count += images.size(0)
            if count >= max_samples:
                break
    emb = np.concatenate(embeddings, axis=0)[:max_samples]
    labels = labels[:max_samples]
    return emb, labels


def _select_gradcam_examples(model, loader, device, num_each: int = 4):
    correct = []
    incorrect = []
    for images, labels, raw_images in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()
        for i in range(images.size(0)):
            item = (images[i], preds[i].item(), labels[i].item(), raw_images[i])
            if preds[i].item() == labels[i].item() and len(correct) < num_each:
                correct.append(item)
            if preds[i].item() != labels[i].item() and len(incorrect) < num_each:
                incorrect.append(item)
            if len(correct) >= num_each and len(incorrect) >= num_each:
                return correct, incorrect
    return correct, incorrect


def make_figures(
    model_path: Path,
    label_mode: str = "binary",
    split: str = "test",
    batch_size: int = 64,
    embedding_samples: int = 2000,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    device: Optional[str] = None,
    backbone: str = "resnet34",
) -> None:
    device = get_device(device)
    ensure_dir(REPORTS_DIR)

    index_path = build_index(label_mode=label_mode)
    label_names = _load_label_names(label_mode)
    num_classes = len(label_names)

    model = build_model(num_classes=num_classes, pretrained=False, backbone=backbone)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    embed_ds = Bbbc041Crops(
        index_path=index_path,
        split=split,
        transform=build_transforms(train=False),
        max_samples=max_samples,
    )
    pin_memory = device.type == "cuda"
    embed_loader = DataLoader(
        embed_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    embedding_samples = min(embedding_samples, len(embed_ds))
    if embedding_samples < 1:
        return
    embeddings, labels = _collect_embeddings(model, embed_loader, device, max_samples=embedding_samples)
    plot_umap(embeddings, labels, label_names, REPORTS_DIR / "umap_embeddings.png")

    raw_transform = T.Compose(
        [
            T.Resize(int(224 * 1.15)),
            T.CenterCrop(224),
        ]
    )
    grad_ds = Bbbc041Crops(
        index_path=index_path,
        split=split,
        transform=build_transforms(train=False),
        raw_transform=raw_transform,
        max_samples=max_samples,
        return_raw=True,
    )
    grad_loader = DataLoader(
        grad_ds,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    correct, incorrect = _select_gradcam_examples(model, grad_loader, device)

    cam = GradCAM(model, model.backbone.layer4)

    for examples, out_name in [(correct, "gradcam_correct.png"), (incorrect, "gradcam_incorrect.png")]:
        images = []
        heatmaps = []
        titles = []
        for img_tensor, pred, label, raw in examples:
            cam_map = cam(img_tensor.unsqueeze(0).to(device), class_idx=pred)
            raw_np = np.array(raw).astype("float32") / 255.0
            images.append(raw_np)
            heatmaps.append(cam_map)
            titles.append(f"pred={label_names[pred]} | true={label_names[label]}")
        if images:
            save_gradcam_grid(images, heatmaps, titles, REPORTS_DIR / out_name)
