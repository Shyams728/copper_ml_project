from importlib import util


def run_pipeline() -> None:
    required_modules = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "lightgbm",
        "category_encoders",
        "imblearn",
        "joblib",
        "openpyxl",
    ]
    missing = [module for module in required_modules if util.find_spec(module) is None]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(
            "Missing required dependencies: {missing}. "
            "Install them with `python -m pip install -r requirements.txt`."
            .format(missing=missing_list)
        )

    from .data import prepare_cleaned_data
    from .training import train_all

    prepare_cleaned_data()
    results = train_all()

    print("\nPipeline completed.")
    print(
        "Regression R2: {regression_score:.4f}\n"
        "Classification Accuracy: {classification_score:.4f}\n"
        "Regression model: {regression_model_path}\n"
        "Classification model: {classification_model_path}"
        .format(**results)
    )


if __name__ == "__main__":
    run_pipeline()
