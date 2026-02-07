# Copper ML Project

End-to-end machine learning workflow for **copper price prediction** (regression) and **lead classification** (classification). The project is organized around Jupyter notebooks that cover exploratory data analysis (EDA), data cleaning, feature engineering, model training, and evaluation.

## Project Goals

- **Predict copper prices** based on structured input features.
- **Classify leads** into outcome categories using supervised learning.
- Provide **iterative notebooks** that document the full workflow from raw data through modeling.

## Repository Contents

### Core Notebooks

- **`copper_project_consolidated.ipynb`**: Consolidated and refactored notebook with block-by-block explanations. This is the recommended starting point.
- **`copper_modeling_final.ipynb`**: End-to-end modeling workflow (original).
- **`copper_price_prediction_and_leads_classification.ipynb`**: Combined regression + classification flow (original).

### Supporting Notebooks

- **`copper_EDA.ipynb`**: Exploratory data analysis (original).
- **`copper_EDA_ML.ipynb`**: EDA with transformations (original).
- **`industrial_copper_modeling_data_cleaning.ipynb`**: Data cleaning focus (original).
- **`industrial_copper_modeling_and_prediction.ipynb`**: Modeling and prediction focus (original).

## How to Use This Project

1. **Start with the consolidated notebook**
   - Open `copper_project_consolidated.ipynb` to follow the full workflow in a single place.

2. **Dive into specific stages**
   - Use the EDA notebooks to understand data distributions and relationships.
   - Use the data-cleaning notebook to follow preprocessing decisions.
   - Use modeling notebooks to compare algorithms and metrics.

3. **Trace the modeling decisions**
   - The original notebooks provide the full history and experimentation process.

## Typical Workflow (Notebook-Driven)

While the exact steps are notebook-specific, the workflow generally follows:

1. **Data ingestion**
2. **EDA and visualization**
3. **Data cleaning and preprocessing**
4. **Feature engineering & transformation**
5. **Model training** (regression + classification)
6. **Model evaluation**
7. **Prediction and interpretation**

## Getting Started

### Prerequisites

- Python 3.8+ (recommended)
- JupyterLab or Jupyter Notebook

Typical ML dependencies (install as needed):

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

> Note: The project does not ship a `requirements.txt`. Install dependencies based on the imports you see in the notebooks.

### Run the Notebooks

```bash
# from the repo root
jupyter lab
```

Then open the notebook(s) you want to run.

## Data

This repository does not include raw datasets. Follow the notebook instructions/cells to locate or load the data required for each workflow.

## Project Structure

```
.
├── README.md
├── copper_project_consolidated.ipynb
├── copper_modeling_final.ipynb
├── copper_price_prediction_and_leads_classification.ipynb
├── copper_EDA.ipynb
├── copper_EDA_ML.ipynb
├── industrial_copper_modeling_data_cleaning.ipynb
└── industrial_copper_modeling_and_prediction.ipynb
```

## Notes

- Notebooks labeled **“original”** reflect earlier iterations and experimentation.
- The consolidated notebook is the most up-to-date, organized version of the workflow.

## License

See `LICENSE` for details.
