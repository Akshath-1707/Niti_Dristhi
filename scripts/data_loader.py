from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def load_csv(filename):
    """
    Load a CSV file from the raw data directory.
    """
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(file_path)


def save_processed_data(df, filename):
    """
    Save processed dataframe.
    """
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)

    return file_path
