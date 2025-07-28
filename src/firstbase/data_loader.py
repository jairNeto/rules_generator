"""Data loader module for the Firstbase AI Challenge."""

import pandas as pd
from pathlib import Path
from typing import Any
import logging
from .config import DEFAULT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader class for handling CSV data loading and basic operations."""

    def __init__(self, data_path: str | None = None):
        """
        Initialize the DataLoader.

        Args:
            data_path: Path to the CSV file
        """
        if data_path is None:
            data_path = "data/raw/retail_store_sales.csv"
        self.data_path = Path(data_path)
        self.data: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV data into a pandas DataFrame.

        Returns:
            Loaded DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(
            f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns"
        )

        return self.data

    def get_data_info(self) -> dict[str, Any]:
        """
        Get basic information about the loaded data.

        Returns:
            Dictionary with data information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "null_counts": self.data.isnull().sum().to_dict(),
            "duplicate_rows": self.data.duplicated().sum(),
        }

        return info

    def get_sample_data(
        self, n: int | None = None, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Get a sample of the data.

        Args:
            n: Number of rows to sample
            random_state: Random state for reproducibility

        Returns:
            Sample DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        sample_size = n or DEFAULT_CONFIG.data.sample_size
        return self.data.sample(sample_size, random_state=random_state)

    def get_column_stats(self, column: str) -> dict[str, Any]:
        """
        Get statistics for a specific column.

        Args:
            column: Column name

        Returns:
            Dictionary with column statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        col_data = self.data[column]
        stats = {
            "dtype": str(col_data.dtype),
            "null_count": col_data.isnull().sum(),
            "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
            "unique_count": col_data.nunique(),
            "unique_percentage": (col_data.nunique() / len(col_data)) * 100,
        }

        # Add descriptive statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update(
                {
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                }
            )

        return stats
