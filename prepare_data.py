import argparse
from pathlib import Path

from copper_ml.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from copper_ml.data import prepare_cleaned_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned copper data.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw Copper_Set.xlsx file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help="Where to write cleaned_copper_data.csv.",
    )
    args = parser.parse_args()

    cleaned_df = prepare_cleaned_data(args.raw_path, args.output_path)
    print(f"Cleaned data saved to {args.output_path} ({cleaned_df.shape[0]:,} rows).")


if __name__ == "__main__":
    main()
