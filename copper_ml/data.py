from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH, ensure_directories


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    df = df.rename(
        columns={
            "customer": "customer_code",
            "country": "country_code",
            "status": "leads",
        }
    )
    return df


def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["item_date", "delivery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    numeric_cols = ["quantity_tons", "thickness", "width", "selling_price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["quantity_tons", "thickness", "width"]:
        if col in df.columns:
            df[col] = df[col].abs()

    categorical_cols = [
        "customer_code",
        "country_code",
        "application",
        "item_type",
        "product_ref",
        "leads",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": np.nan})
            df[col] = df[col].str.replace(r"\.0$", "", regex=True)

    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["quantity_tons", "thickness", "width"]
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    categorical_cols = [
        "customer_code",
        "country_code",
        "application",
        "item_type",
        "product_ref",
        "leads",
    ]
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna("missing")

    return df


def _filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "quantity_tons" in df.columns:
        df = df[df["quantity_tons"] > 0]
    df = df.drop_duplicates()
    return df


def prepare_cleaned_data(
    raw_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    ensure_directories()
    raw_path = raw_path or RAW_DATA_PATH
    output_path = output_path or PROCESSED_DATA_PATH

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    df = pd.read_excel(raw_path)
    df = _standardize_columns(df)
    df = _clean_types(df)
    df = _impute_missing(df)
    df = _filter_rows(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
