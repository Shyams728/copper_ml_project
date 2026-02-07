from .data import prepare_cleaned_data
from .training import train_all


def run_pipeline() -> None:
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
