# Copper ML Project

End-to-end machine learning workflow for **copper price prediction** (regression) and **lead classification** (classification). The project includes notebooks for EDA and modeling plus a Streamlit app for interactive exploration and predictions.

## Project Goals

- **Predict copper prices** based on structured input features.
- **Classify leads** into outcome categories using supervised learning.
- Provide **iterative notebooks** that document the full workflow from raw data through modeling.

## Repository Contents

### Core Notebooks

- **`notebooks/copper_project_consolidated.ipynb`**: Consolidated and refactored notebook with block-by-block explanations. This is the recommended starting point.
- **`notebooks/copper_modeling_final.ipynb`**: End-to-end modeling workflow (original).
- **`notebooks/copper_price_prediction_and_leads_classification.ipynb`**: Combined regression + classification flow (original).

### Supporting Notebooks

- **`notebooks/copper_EDA.ipynb`**: Exploratory data analysis (original).
- **`notebooks/copper_EDA_ML.ipynb`**: EDA with transformations (original).
- **`notebooks/industrial_copper_modeling_data_cleaning.ipynb`**: Data cleaning focus (original).
- **`notebooks/industrial_copper_modeling_and_prediction.ipynb`**: Modeling and prediction focus (original).

### Source Code

- **`copper_ml/`**: Package that holds data prep, training, and pipeline helpers.
- **`prepare_data.py`**: CLI to clean raw data and write `data/processed/cleaned_copper_data.csv`.
- **`train_models.py`**: CLI to train and save regression + classification models.
- **`streamlit_app.py`**: Streamlit interface for analytics and predictions.

## Project Structure

```
.
├── copper_ml/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── pipeline.py
│   └── training.py
├── data/
│   ├── raw/
│   │   └── Copper_Set.xlsx
│   └── processed/
│       └── cleaned_copper_data.csv
├── assets/
│   └── shubham-dhage-AC4Q1uLRKd4-unsplash.jpg
├── model/
├── notebooks/
│   ├── copper_project_consolidated.ipynb
│   ├── copper_modeling_final.ipynb
│   ├── copper_price_prediction_and_leads_classification.ipynb
│   ├── copper_EDA.ipynb
│   ├── copper_EDA_ML.ipynb
│   ├── industrial_copper_modeling_data_cleaning.ipynb
│   └── industrial_copper_modeling_and_prediction.ipynb
├── prepare_data.py
├── train_models.py
└── streamlit_app.py
```

## Getting Started

### Prerequisites

- Python 3.8+ (recommended)
- JupyterLab or Jupyter Notebook (for notebook workflows)

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

### Run the End-to-End Pipeline

```bash
python -m copper_ml.pipeline
```

### Prepare Data + Train Models (Streamlit)

The Streamlit app expects cleaned data and trained model artifacts under `model/`.

```bash
python prepare_data.py
python train_models.py
```

### Launch Streamlit

```bash
streamlit run streamlit_app.py
```

### Run the Notebooks

```bash
# from the repo root
jupyter lab
```

Then open the notebook(s) you want to run.

## Data

The `data/raw/Copper_Set.xlsx` dataset is included in the repository root under the data folder. The `prepare_data.py` script uses it to generate `data/processed/cleaned_copper_data.csv`.

## Notes

- Notebooks labeled **“original”** reflect earlier iterations and experimentation.
- The consolidated notebook is the most up-to-date, organized version of the workflow.

## License

See `LICENSE` for details.
