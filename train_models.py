
import os
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
import warnings

# Suppress Version/Deprecation warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath='model/cleaned_copper_data.csv'):
    print("Loading data...")
    if not os.path.exists(filepath):
        from prepare_data import DEFAULT_RAW_PATH, prepare_cleaned_data

        print(f"Cleaned data not found at {filepath}.")
        print(f"Generating cleaned data from {DEFAULT_RAW_PATH}...")
        prepare_cleaned_data(DEFAULT_RAW_PATH, filepath)

    df = pd.read_csv(filepath)
    
    # Ensure correct types
    # Based on project_artifacts.json inspection
    # categorical_cols = ['item_type', 'application', 'country_code']
    # numerical_cols = ['quantity_tons', 'thickness', 'width', 'selling_price']
    
    # Handling potential bad data in quantity_tons if it was read as object (csv sometimes does that)
    # Reconstruct item_type from one-hot encoded columns
    item_type_cols = ['item_type_PL', 'item_type_S', 'item_type_W', 'item_type_WI']
    
    def get_item_type(row):
        for col in item_type_cols:
            if col in row and row[col] == 1:
                return col.replace('item_type_', '')
        return 'Others' # Default for all-zeros (implicit 'Others', 'IPL', 'SLAWR' etc)

    # Check if encoded cols exist
    existing_type_cols = [c for c in item_type_cols if c in df.columns]
    if existing_type_cols:
        df['item_type'] = df.apply(get_item_type, axis=1)
    else:
        # Fallback if 'item_type' exists (sanity check)
        if 'item_type' not in df.columns:
            # Should not happen based on analysis, but safe fallback
            df['item_type'] = 'Others'

    df['item_type'] = df['item_type'].astype(str)
    df['application'] = df['application'].astype(str)
    df['country_code'] = df['country_code'].astype(str)
    
    # Ensure quantity_tons is numeric (handle errors)
    df['quantity_tons'] = pd.to_numeric(df['quantity_tons'], errors='coerce')
    df = df.dropna(subset=['quantity_tons']) # Drop bad quantity rows
    
    # Filter for Regression (Selling Price)
    # Remove rows where selling_price is <= 0 or missing
    df_reg = df[df['selling_price'] > 0].copy()
    # Log transform target for better distribution (common in this domain)
    # We will predict log(price) then exp() it, or just predict price. 
    # Let's keep it simple: Predict Price, but log transform features.
    
    return df, df_reg

def create_preprocessor():
    # Log transformer logic
    # Use np.log1p directly for pickling support
    log_transformer = FunctionTransformer(np.log1p, validate=False)
    
    # Numerical pipeline: Impute -> Log -> Scale
    # We apply log to skewed columns: quantity_tons, thickness
    skewed_cols = ['quantity_tons', 'thickness']
    normal_cols = ['width'] # width usually less skewed or just scale it
    
    num_skewed_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])
    
    num_normal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute -> OneHot
    cat_cols = ['item_type', 'application', 'country_code']
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_skewed', num_skewed_pipe, skewed_cols),
            ('num_normal', num_normal_pipe, normal_cols),
            ('cat', cat_pipe, cat_cols)
        ]
    )
    return preprocessor

def train_regression_model(df):
    print("\n--- Training Regression Model (Selling Price) ---")
    X = df[['quantity_tons', 'thickness', 'width', 'item_type', 'application', 'country_code']]
    y = df['selling_price']
    
    # Remove crazy outliers for training stability
    # e.g. selling price > 100000 or < 10
    mask = (y > 10) & (y < 100000)
    X = X[mask]
    y = y[mask]
    
    # Log transform target? 
    # Let's simple train on y, but XGBoost handles non-linear well.
    # Actually, price is usually log-normal. Let's predict log(price) for better results?
    # For simplicity of integration, let's predict Price directly, but use log-transformed features.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = create_preprocessor()
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Regression Test R2 Score: {score:.4f}")
    
    return model

def train_classification_model(df):
    print("\n--- Training Classification Model (Lead Status) ---")
    # Filter for Won vs Lost only for binary classification
    # 'leads' column
    # Check values
    valid_status = ['Won', 'Lost']
    df_cls = df[df['leads'].isin(valid_status)].copy()
    
    # Map target
    df_cls['target'] = df_cls['leads'].apply(lambda x: 1 if x == 'Won' else 0)
    
    X = df_cls[['quantity_tons', 'thickness', 'width', 'item_type', 'application', 'country_code']]
    y = df_cls['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = create_preprocessor()
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Classification Test Accuracy: {score:.4f}")
    
    return model

if __name__ == "__main__":
    os.makedirs('model', exist_ok=True)
    df, df_reg = load_and_prepare_data()
    
    # Train and Save Regression
    reg_model = train_regression_model(df_reg)
    joblib.dump(reg_model, 'model/pipeline_regression_model.pkl')
    print("Saved model/pipeline_regression_model.pkl")
    
    # Train and Save Classification
    cls_model = train_classification_model(df)
    joblib.dump(cls_model, 'model/pipeline_classification_model.pkl')
    print("Saved model/pipeline_classification_model.pkl")
    
    print("\nAll models retrained and saved as pipelines!")
