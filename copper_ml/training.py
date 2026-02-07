import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import joblib

from .config import CLS_MODEL_PATH, PROCESSED_DATA_PATH, REG_MODEL_PATH, ensure_directories
from .data import prepare_cleaned_data


def load_and_prepare_data(filepath: str | None = None):
    filepath = filepath or str(PROCESSED_DATA_PATH)
    if not PROCESSED_DATA_PATH.exists():
        prepare_cleaned_data()

    df = pd.read_csv(filepath)

    item_type_cols = ["item_type_PL", "item_type_S", "item_type_W", "item_type_WI"]

    def get_item_type(row):
        for col in item_type_cols:
            if col in row and row[col] == 1:
                return col.replace("item_type_", "")
        return "Others"

    existing_type_cols = [c for c in item_type_cols if c in df.columns]
    if existing_type_cols:
        df["item_type"] = df.apply(get_item_type, axis=1)
    else:
        if "item_type" not in df.columns:
            df["item_type"] = "Others"

    df["item_type"] = df["item_type"].astype(str)
    df["application"] = df["application"].astype(str)
    df["country_code"] = df["country_code"].astype(str)

    df["quantity_tons"] = pd.to_numeric(df["quantity_tons"], errors="coerce")
    df = df.dropna(subset=["quantity_tons"])

    df_reg = df[df["selling_price"] > 0].copy()

    return df, df_reg


def create_preprocessor():
    log_transformer = FunctionTransformer(np.log1p, validate=False)

    skewed_cols = ["quantity_tons", "thickness"]
    normal_cols = ["width"]

    num_skewed_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", log_transformer),
            ("scaler", StandardScaler()),
        ]
    )

    num_normal_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_cols = ["item_type", "application", "country_code"]
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_skewed", num_skewed_pipe, skewed_cols),
            ("num_normal", num_normal_pipe, normal_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor


def train_regression_model(df):
    X = df[
        [
            "quantity_tons",
            "thickness",
            "width",
            "item_type",
            "application",
            "country_code",
        ]
    ]
    y = df["selling_price"]

    mask = (y > 10) & (y < 100000)
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = create_preprocessor()

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    objective="reg:squarederror", n_estimators=100, random_state=42
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, score


def train_classification_model(df):
    valid_status = ["Won", "Lost"]
    df_cls = df[df["leads"].isin(valid_status)].copy()

    df_cls["target"] = df_cls["leads"].apply(lambda x: 1 if x == "Won" else 0)

    X = df_cls[
        [
            "quantity_tons",
            "thickness",
            "width",
            "item_type",
            "application",
            "country_code",
        ]
    ]
    y = df_cls["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = create_preprocessor()

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, score


def train_all():
    ensure_directories()
    df, df_reg = load_and_prepare_data()

    reg_model, reg_score = train_regression_model(df_reg)
    joblib.dump(reg_model, REG_MODEL_PATH)

    cls_model, cls_score = train_classification_model(df)
    joblib.dump(cls_model, CLS_MODEL_PATH)

    return {
        "regression_score": reg_score,
        "classification_score": cls_score,
        "regression_model_path": REG_MODEL_PATH,
        "classification_model_path": CLS_MODEL_PATH,
    }
