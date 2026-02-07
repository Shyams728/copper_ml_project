
import pickle
import joblib
import os
import sys

def inspect_model(path):
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found")
        return

    try:
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
        except:
            model = joblib.load(path)
            
        print(f"Type: {type(model)}")
        if hasattr(model, 'steps'):
            print("Is Pipeline: Yes")
            print("Steps:", [s[0] for s in model.steps])
        else:
            print("Is Pipeline: No")
            
    except Exception as e:
        print(f"Error loading: {e}")

base_path = r"d:\data_science\industrial copper modeling\model"
files = [
    "best_classification_model.pkl",
    "best_regression_model.pkl", 
    "final_classification_model.pkl",
    "final_regression_model.pkl"
]

for f in files:
    inspect_model(os.path.join(base_path, f))
