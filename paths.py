
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"

def project_relative_path(filepath: Path) -> str:
    """Return filepath relative to PROJECT_DIR"""
    return os.path.relpath(filepath.resolve(), PROJECT_DIR.resolve())