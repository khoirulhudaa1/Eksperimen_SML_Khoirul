from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "breast_cancer_raw" / "breast_cancer_raw.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "preprocessing"
    / "breast_cancer_preprocessing"
    / "breast_cancer_processed.csv"
)
DEFAULT_METADATA = (
    PROJECT_DIR
    / "preprocessing"
    / "breast_cancer_preprocessing"
    / "preprocessing_metadata.json"
)
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = (
        dataframe.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return dataframe


def load_data(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset raw tidak ditemukan: {input_path}")
    return pd.read_csv(input_path)


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = normalize_column_names(dataframe)
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak tersedia.")

    numeric_columns = [column for column in dataframe.columns if column != TARGET_COLUMN]
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    dataframe[TARGET_COLUMN] = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    dataframe = dataframe.dropna().reset_index(drop=True)
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)
    return dataframe


def preprocess_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = clean_data(dataframe)
    feature_columns = [column for column in cleaned.columns if column != TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned[feature_columns],
        cleaned[TARGET_COLUMN],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=cleaned[TARGET_COLUMN],
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=feature_columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=feature_columns,
        index=x_test.index,
    )

    train_processed = x_train_scaled.copy()
    train_processed[TARGET_COLUMN] = y_train
    train_processed["split"] = "train"

    test_processed = x_test_scaled.copy()
    test_processed[TARGET_COLUMN] = y_test
    test_processed["split"] = "test"

    processed = (
        pd.concat([train_processed, test_processed], axis=0)
        .sort_index()
        .reset_index(drop=True)
    )

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "random_state": RANDOM_STATE,
        "train_rows": int((processed["split"] == "train").sum()),
        "test_rows": int((processed["split"] == "test").sum()),
        "total_rows": int(processed.shape[0]),
        "total_features": len(feature_columns),
        "scaler": "StandardScaler fitted on train split only",
    }
    return processed, metadata


def save_processed_data(
    processed: pd.DataFrame,
    metadata: dict,
    output_path: str | Path = DEFAULT_OUTPUT,
    metadata_path: str | Path = DEFAULT_METADATA,
) -> None:
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_preprocessing(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    metadata_path: str | Path = DEFAULT_METADATA,
) -> pd.DataFrame:
    dataframe = load_data(input_path)
    processed, metadata = preprocess_data(dataframe)
    save_processed_data(processed, metadata, output_path, metadata_path)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Breast Cancer dataset untuk proyek MSML."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path dataset raw.")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Path output dataset processed."
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Path output metadata preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = run_preprocessing(args.input, args.output, args.metadata)
    print(f"Preprocessing selesai. Shape dataset: {processed.shape}")
    print(f"Output: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
