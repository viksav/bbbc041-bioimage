import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from .config import EXTERNAL_DIR, NIH_MALARIA_URL
from .utils import ensure_dir


def download_nih_malaria(raw_dir: Path = EXTERNAL_DIR, url: str = NIH_MALARIA_URL) -> Path:
    ensure_dir(raw_dir)
    zip_path = raw_dir / "cell_images.zip"
    if not zip_path.exists():
        print(f"Downloading {url} -> {zip_path}")
        urlretrieve(url, zip_path)
    return zip_path


def extract_nih_malaria(zip_path: Path, raw_dir: Path = EXTERNAL_DIR) -> Path:
    extract_dir = raw_dir / "cell_images"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return extract_dir
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return extract_dir


def prepare_nih_malaria() -> Path:
    zip_path = download_nih_malaria()
    return extract_nih_malaria(zip_path)
