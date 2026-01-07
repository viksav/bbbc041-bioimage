import csv
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from urllib.request import urlretrieve

from .config import BBBC041_URL, PROCESSED_DIR, RAW_DIR
from .utils import ensure_dir, save_json


BINARY_LABEL_MAP = {
    "red blood cell": 0,
    "trophozoite": 1,
    "ring": 1,
    "schizont": 1,
    "gametocyte": 1,
}

MULTICLASS_LABELS = [
    "red blood cell",
    "trophozoite",
    "ring",
    "schizont",
    "gametocyte",
    "leukocyte",
]

IGNORE_CATEGORIES = {"difficult"}


@dataclass
class Annotation:
    image_id: str
    image_path: Path
    label_id: int
    label_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    split: str


def download_bbbc041(raw_dir: Path = RAW_DIR, url: str = BBBC041_URL) -> Path:
    ensure_dir(raw_dir)
    zip_path = raw_dir / "malaria.zip"
    if not zip_path.exists():
        print(f"Downloading {url} -> {zip_path}")
        urlretrieve(url, zip_path)
    return zip_path


def extract_bbbc041(zip_path: Path, raw_dir: Path = RAW_DIR) -> Path:
    extract_dir = raw_dir / "malaria"
    if extract_dir.exists() and (extract_dir / "training.json").exists():
        return extract_dir
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return extract_dir


def _load_training_json(extract_dir: Path) -> List[dict]:
    training_json = extract_dir / "training.json"
    with open(training_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_splits(image_ids: List[str], seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    split_map = {}
    for image_id in ids[:n_train]:
        split_map[image_id] = "train"
    for image_id in ids[n_train : n_train + n_val]:
        split_map[image_id] = "val"
    for image_id in ids[n_train + n_val :]:
        split_map[image_id] = "test"
    return split_map


def _label_map(label_mode: str) -> Dict[str, int]:
    if label_mode == "binary":
        return {"uninfected": 0, "infected": 1}
    if label_mode == "multiclass":
        return {name: idx for idx, name in enumerate(MULTICLASS_LABELS)}
    raise ValueError(f"Unknown label_mode: {label_mode}")


def _map_label(label_mode: str, category: str) -> Optional[Tuple[int, str]]:
    if category in IGNORE_CATEGORIES:
        return None
    if label_mode == "binary":
        if category not in BINARY_LABEL_MAP:
            return None
        return BINARY_LABEL_MAP[category], ("infected" if category != "red blood cell" else "uninfected")
    if label_mode == "multiclass":
        if category not in MULTICLASS_LABELS:
            return None
        return MULTICLASS_LABELS.index(category), category
    raise ValueError(f"Unknown label_mode: {label_mode}")


def build_index(
    label_mode: str = "binary",
    seed: int = 42,
    force: bool = False,
) -> Path:
    ensure_dir(PROCESSED_DIR)
    index_path = PROCESSED_DIR / f"index_{label_mode}.csv"
    label_map_path = PROCESSED_DIR / f"label_map_{label_mode}.json"
    splits_path = PROCESSED_DIR / "splits.json"

    if index_path.exists() and not force:
        return index_path

    zip_path = download_bbbc041()
    extract_dir = extract_bbbc041(zip_path)
    data = _load_training_json(extract_dir)

    image_ids = [Path(item["image"]["pathname"]).stem for item in data]
    split_map = _make_splits(sorted(set(image_ids)), seed)

    annotations: List[Annotation] = []
    for item in data:
        image_path = extract_dir / item["image"]["pathname"].lstrip("/")
        image_id = Path(image_path).stem
        split = split_map[image_id]
        for obj in item["objects"]:
            category = obj["category"]
            mapped = _map_label(label_mode, category)
            if mapped is None:
                continue
            label_id, label_name = mapped
            bbox = obj["bounding_box"]
            xmin = int(bbox["minimum"]["c"])
            ymin = int(bbox["minimum"]["r"])
            xmax = int(bbox["maximum"]["c"])
            ymax = int(bbox["maximum"]["r"])
            annotations.append(
                Annotation(
                    image_id=image_id,
                    image_path=image_path,
                    label_id=label_id,
                    label_name=label_name,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    split=split,
                )
            )

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_id",
                "image_path",
                "label_id",
                "label_name",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "split",
            ]
        )
        for ann in annotations:
            writer.writerow(
                [
                    ann.image_id,
                    str(ann.image_path),
                    ann.label_id,
                    ann.label_name,
                    ann.xmin,
                    ann.ymin,
                    ann.xmax,
                    ann.ymax,
                    ann.split,
                ]
            )

    save_json(_label_map(label_mode), label_map_path)
    save_json(split_map, splits_path)

    return index_path


def load_index(index_path: Path) -> List[Annotation]:
    rows: List[Annotation] = []
    with open(index_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Annotation(
                    image_id=row["image_id"],
                    image_path=Path(row["image_path"]),
                    label_id=int(row["label_id"]),
                    label_name=row["label_name"],
                    xmin=int(row["xmin"]),
                    ymin=int(row["ymin"]),
                    xmax=int(row["xmax"]),
                    ymax=int(row["ymax"]),
                    split=row["split"],
                )
            )
    return rows


class Bbbc041Crops(Dataset):
    def __init__(
        self,
        index_path: Path,
        split: str,
        transform=None,
        raw_transform=None,
        margin: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        return_raw: bool = False,
    ) -> None:
        self.split = split
        self.transform = transform
        self.raw_transform = raw_transform
        self.margin = margin
        self.return_raw = return_raw

        rows = [row for row in load_index(index_path) if row.split == split]
        if max_samples is not None and max_samples > 0 and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, max_samples)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row.image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        crop = self._crop_with_margin(image, row)
        raw = None
        if self.return_raw:
            raw_image = crop.copy()
            if self.raw_transform is not None:
                raw_image = self.raw_transform(raw_image)
            raw = np.array(raw_image)

        if self.transform is not None:
            crop = self.transform(crop)

        if self.return_raw:
            return crop, row.label_id, raw
        return crop, row.label_id

    def _crop_with_margin(self, image: Image.Image, row: Annotation) -> Image.Image:
        width, height = image.size
        box_w = max(1, row.xmax - row.xmin)
        box_h = max(1, row.ymax - row.ymin)
        pad_x = int(box_w * self.margin)
        pad_y = int(box_h * self.margin)
        xmin = max(0, row.xmin - pad_x)
        ymin = max(0, row.ymin - pad_y)
        xmax = min(width, row.xmax + pad_x)
        ymax = min(height, row.ymax + pad_y)
        return image.crop((xmin, ymin, xmax, ymax))
