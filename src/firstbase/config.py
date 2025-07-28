"""Configuration module for the Firstbase AI Challenge."""

from dataclasses import dataclass, field


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""

    model: str = "gpt-4.1-2025-04-14"
    temperature: float = 0.0


@dataclass
class DataConfig:
    """Data processing configuration."""

    sample_size: int = 20
    output_dir: str = "data/processed"
    rules_filename: str = "cleaning_rules.json"
    cleaned_data_filename: str = "cleaned_data.csv"
    metrics_filename: str = "cleaning_metrics.json"
    data_description: str = """
    The Dirty Retail Store Sales dataset contains 12,575 rows of synthetic data representing sales transactions from a retail store.
    The dataset includes eight product categories with 25 items per category, each having static prices.
    It is designed to simulate real-world sales data, including intentional "dirtiness" such as missing or inconsistent values.
    This dataset is suitable for practicing data cleaning, exploratory data analysis (EDA), and feature engineering.
    """


@dataclass
class AppConfig:
    """Main application configuration."""

    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    data: DataConfig = field(default_factory=DataConfig)


# Default configuration
DEFAULT_CONFIG = AppConfig()

