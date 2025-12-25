import pandas as pd
from typing import Optional
import os


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV file."""
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise ValueError(f"Error saving CSV: {str(e)}")


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)
