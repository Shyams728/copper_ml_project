import warnings

from copper_ml.training import train_all


warnings.filterwarnings("ignore")


def main() -> None:
    results = train_all()
    print("\nAll models retrained and saved as pipelines!")
    print(
        "Regression R2: {regression_score:.4f}\n"
        "Classification Accuracy: {classification_score:.4f}\n"
        "Regression model: {regression_model_path}\n"
        "Classification model: {classification_model_path}".format(**results)
    )


if __name__ == "__main__":
    main()
