from pathlib import Path

import torch
from PIL import Image

from bioimage.data import Bbbc041Crops
from bioimage.transforms import build_transforms


def test_can_load_one_batch(tmp_path: Path) -> None:
    image_path = tmp_path / "cell.png"
    Image.new("RGB", (64, 64), color=(128, 64, 32)).save(image_path)

    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "image_id,image_path,label_id,label_name,xmin,ymin,xmax,ymax,split\n"
        f"img_1,{image_path},0,uninfected,4,4,60,60,train\n",
        encoding="utf-8",
    )

    ds = Bbbc041Crops(
        index_path=index_path,
        split="train",
        transform=build_transforms(train=False, image_size=32),
    )

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert label == 0
