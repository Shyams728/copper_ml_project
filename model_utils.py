import joblib
import pandas as pd

from copper_ml.config import CLS_MODEL_PATH, REG_MODEL_PATH


def load_artifacts():
    """Load the trained models (pipelines)."""
    artifacts = {}

    artifacts["reg_model_path"] = str(REG_MODEL_PATH)
    if REG_MODEL_PATH.exists():
        try:
            artifacts["reg_model"] = joblib.load(REG_MODEL_PATH)
        except Exception as e:
            print(f"Error loading regression model: {e}")
            artifacts["reg_model"] = None
            artifacts["reg_model_error"] = str(e)
    else:
        print(f"Regression model not found at {REG_MODEL_PATH}")
        artifacts["reg_model"] = None
        artifacts["reg_model_error"] = f"File not found at {REG_MODEL_PATH}"

    artifacts["cls_model_path"] = str(CLS_MODEL_PATH)
    if CLS_MODEL_PATH.exists():
        try:
            artifacts["cls_model"] = joblib.load(CLS_MODEL_PATH)
        except Exception as e:
            print(f"Error loading classification model: {e}")
            artifacts["cls_model"] = None
            artifacts["cls_model_error"] = str(e)
    else:
        print(f"Classification model not found at {CLS_MODEL_PATH}")
        artifacts["cls_model"] = None
        artifacts["cls_model_error"] = f"File not found at {CLS_MODEL_PATH}"

    return artifacts


def _prepare_input_df(data):
    """Convert input dictionary to DataFrame with correct types."""
    expected_cols = [
        "quantity_tons",
        "thickness",
        "width",
        "item_type",
        "application",
        "country_code",
    ]
    df = pd.DataFrame([data]).reindex(columns=expected_cols)

    df["quantity_tons"] = pd.to_numeric(df["quantity_tons"], errors="coerce")
    df["thickness"] = pd.to_numeric(df["thickness"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")

    df["item_type"] = df["item_type"].astype(str)
    df["application"] = df["application"].astype(str)
    df["country_code"] = df["country_code"].astype(str)

    return df


def predict_selling_price(data, artifacts):
    """Predict selling price using regression model."""
    model = artifacts.get("reg_model")
    if not model:
        return None

    try:
        df = _prepare_input_df(data)
        prediction = model.predict(df)[0]
        return max(0, prediction)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None


def predict_status(data, artifacts):
    """Predict lead status (Won/Lost)."""
    model = artifacts.get("cls_model")
    if not model:
        return None, None

    try:
        df = _prepare_input_df(data)
        pred_class = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]

        return pred_class, probabilities
    except Exception as e:
        print(f"Classification Error: {e}")
        return None, None
