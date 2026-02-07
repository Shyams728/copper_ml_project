
import joblib
import pandas as pd
import numpy as np
import os
import xgboost
from sklearn.ensemble import RandomForestClassifier

# Define paths
# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
REG_MODEL_PATH = os.path.join(MODEL_DIR, 'pipeline_regression_model.pkl')
CLS_MODEL_PATH = os.path.join(MODEL_DIR, 'pipeline_classification_model.pkl')

def load_artifacts():
    """Load the trained models (pipelines)"""
    artifacts = {}
    
    # Load Regression Model
    artifacts['reg_model_path'] = REG_MODEL_PATH
    if os.path.exists(REG_MODEL_PATH):
        try:
            artifacts['reg_model'] = joblib.load(REG_MODEL_PATH)
        except Exception as e:
            print(f"Error loading regression model: {e}")
            artifacts['reg_model'] = None
            artifacts['reg_model_error'] = str(e)
    else:
         print(f"Regression model not found at {REG_MODEL_PATH}")
         artifacts['reg_model'] = None
         artifacts['reg_model_error'] = f"File not found at {REG_MODEL_PATH}"

    # Load Classification Model
    artifacts['cls_model_path'] = CLS_MODEL_PATH
    if os.path.exists(CLS_MODEL_PATH):
        try:
            artifacts['cls_model'] = joblib.load(CLS_MODEL_PATH)
        except Exception as e:
            print(f"Error loading classification model: {e}")
            artifacts['cls_model'] = None
            artifacts['cls_model_error'] = str(e)
    else:
         print(f"Classification model not found at {CLS_MODEL_PATH}")
         artifacts['cls_model'] = None
         artifacts['cls_model_error'] = f"File not found at {CLS_MODEL_PATH}"
            
    return artifacts

def _prepare_input_df(data):
    """Convert input dictionary to DataFrame with correct types"""
    # Expected columns by the pipeline
    # num_skewed: quantity_tons, thickness
    # num_normal: width
    # cat: item_type, application, country_code
    
    df = pd.DataFrame([data])
    
    # Ensure types match training
    df['quantity_tons'] = pd.to_numeric(df['quantity_tons'], errors='coerce')
    df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
    df['width'] = pd.to_numeric(df['width'], errors='coerce')
    
    df['item_type'] = df['item_type'].astype(str)
    df['application'] = df['application'].astype(str)
    df['country_code'] = df['country_code'].astype(str)
    
    return df

def predict_selling_price(data, artifacts):
    """Predict selling price using regression model"""
    model = artifacts.get('reg_model')
    if not model:
        return None
        
    try:
        df = _prepare_input_df(data)
        prediction = model.predict(df)[0]
        # If we log transformed target during training, we'd exp() here. 
        # But in train_models.py I decided to train on raw price (with log-features).
        # So return raw prediction.
        return max(0, prediction) # Ensure no negative prices
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

def predict_status(data, artifacts):
    """Predict lead status (Won/Lost)"""
    model = artifacts.get('cls_model')
    if not model:
        return None, None
        
    try:
        df = _prepare_input_df(data)
        pred_class = model.predict(df)[0] # 1 or 0
        probabilities = model.predict_proba(df)[0] # [prob_0, prob_1]
        
        return pred_class, probabilities
    except Exception as e:
        print(f"Classification Error: {e}")
        return None, None
