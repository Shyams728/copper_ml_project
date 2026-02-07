from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Copper_Set.xlsx"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_copper_data.csv"
MODEL_DIR = BASE_DIR / "model"
REG_MODEL_PATH = MODEL_DIR / "pipeline_regression_model.pkl"
CLS_MODEL_PATH = MODEL_DIR / "pipeline_classification_model.pkl"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
