from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Main folders
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# Dataset
DATA_FILE = DATA_DIR / "Ecommerce.csv"


# Cleaned dataset
CLEANED_DATA_FILE = DATA_DIR / "cleaned_ecommerce.csv"