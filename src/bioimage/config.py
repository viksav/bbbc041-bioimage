from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "bbbc041"
PROCESSED_DIR = DATA_DIR / "processed" / "bbbc041"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

BBBC041_URL = "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"
NIH_MALARIA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
EXTERNAL_DIR = DATA_DIR / "external" / "nih_malaria"
