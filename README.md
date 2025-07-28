# Firstbase AI Challenge - Data Cleaning Rule Generator

This project implements an automated data cleaning rule generator that uses AI to analyze messy datasets and generate human-readable cleaning rules. The system automatically learns patterns from noisy data and applies interpretable rules to clean and standardize datasets.

## Features

- **Data Loading**: Simple CSV data loader with comprehensive analysis
- **AI Rule Generation**: OpenAI-powered rule generation using GPT models
- **Rule Application**: Apply generated rules to clean datasets with comprehensive logging
- **Anomaly Detection**: Automatic flagging of suspicious data patterns
- **Data Writing**: JSON output for rules and cleaned data
- **Environment Configuration**: Secure API key management with .env files
- **Configuration Management**: Centralized config for non-secret parameters
- **Testing**: Comprehensive unit tests for rule logic and validation
- **Modern Python**: Uses Python 3.11+ with modern typing

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure OpenAI API**:
   - Copy your OpenAI API key to the `.env` file:
   ```bash
   # Edit .env and add your OpenAI API key
   echo "OPENAI_API_KEY=your_actual_key_here" > .env
   ```

3. **Run the EDA script** to understand your data:
   ```bash
   uv run python eda.py
   ```

## Usage

### Complete Workflow

1. **Generate Cleaning Rules**:
   ```bash
   uv run python generate_rules.py data/raw/retail_store_sales.csv
   ```

2. **Apply Rules to Data**:
   ```bash
   uv run python apply_rules.py data/raw/retail_store_sales.csv --show-summary
   ```

3. **Run Tests** (optional):
   ```bash
   uv run pytest tests/ -v
   ```

### Generate Cleaning Rules

```bash
# Using environment variable for API key
uv run python generate_rules.py data/raw/retail_store_sales.csv

# Or specify API key directly
uv run python generate_rules.py data/raw/retail_store_sales.csv --api-key your_key_here

# With custom parameters
uv run python generate_rules.py data/raw/retail_store_sales.csv \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --sample-size 100 \
  --output-dir data/processed
```

### Apply Rules to Data

```bash
# Apply rules with summary
uv run python apply_rules.py data/raw/retail_store_sales.csv --show-summary

# Custom rules file and output
uv run python apply_rules.py data/raw/retail_store_sales.csv \
  --rules-path data/processed/cleaning_rules.json \
  --output-dir data/processed \
  --output-filename my_cleaned_data.csv
```

### Command Line Options

#### Generate Rules
- `csv_path`: Path to the CSV file to analyze
- `--api-key`: OpenAI API key (optional if set in .env)
- `--model`: OpenAI model to use (default: gpt-4o-mini)
- `--temperature`: Generation temperature (default: 0.0)
- `--sample-size`: Number of rows to sample for analysis (default: 20)
- `--output-dir`: Output directory for results (default: data/processed)

#### Apply Rules
- `csv_path`: Path to the input CSV file
- `--rules-path`: Path to the JSON rules file (default: data/processed/cleaning_rules.json)
- `--output-dir`: Output directory for results (default: data/processed)
- `--output-filename`: Output filename for cleaned data (default: cleaned_data.csv)
- `--log-filename`: Output filename for cleaning log (default: cleaning_log.json)
- `--show-summary`: Show a summary of applied rules

## Configuration

### Environment Variables (.env)
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Application Configuration (src/firstbase/config.py)
```python
# Default OpenAI settings
model = "gpt-4o-mini"
temperature = 0.0

# Default data processing settings
sample_size = 20
output_dir = "data/processed"
```

## Output

The system generates:

1. **Cleaning Rules** (`cleaning_rules.json`): Human-readable rules with confidence scores and rule types
2. **Cleaned Data** (`cleaned_data.csv`): Processed dataset with applied rules
3. **Cleaning Log** (`cleaning_log.json`): Detailed log of all applied rules and changes
4. **Flag Columns**: New columns for anomaly detection (e.g., `Item_flag_flag_missing_item`)

## Project Structure

```
firstbase/
├── src/firstbase/
│   ├── __init__.py
│   ├── config.py              # Application configuration
│   ├── data_loader.py         # Data loading and analysis
│   ├── cleaned_rule_generator.py  # AI rule generation
│   ├── rule_applier.py        # Rule application logic
│   ├── data_writer.py         # Output handling
├── tests/
│   └── test_rule_applier.py   # Unit tests for rule application
├── data/
│   ├── raw/                   # Input CSV files
│   └── processed/             # Generated outputs
├── eda.py                     # Exploratory data analysis
├── generate_rules.py          # Rule generation script
├── apply_rules.py             # Rule application script
├── .env                       # Environment variables (API key only)
└── pyproject.toml            # Project configuration
```

## Rule Format

The system uses a structured rule format with explicit rule types:

```json
{
  "rule_id": "normalize_phone",
  "description": "Normalize phone numbers to +1-XXX-XXX-XXXX format",
  "confidence": 0.94,
  "pattern": "(\\d{3})[. -]?(\\d{3})[. -]?(\\d{4})",
  "replacement": "+1-\\1-\\2-\\3",
  "rule_type": "transformation",
  "columns": ["phone_number"],
  "reasoning": "Standardize phone number format for consistency"
}
```

## Rule Types

The system supports four types of rules:

1. **Transformation Rules** (`transformation`): Modify data values (e.g., formatting, normalization)
2. **Imputation Rules** (`imputation`): Fill missing values with calculated or default values
3. **Anomaly Detection Rules** (`anomaly_flag`): Flag suspicious or outlier values for review
4. **Format String Rules** (`format_string`): Apply special formatting like rounding numbers

## Results

### Data Quality Improvements

The system successfully processed the retail store sales dataset:

- **Missing Values**: Reduced from 7,229 to 0 (100% improvement)
- **Dataset Size**: 12,575 rows, 20 columns (9 new flag columns)
- **Anomaly Detection**: 9 different anomaly flags applied
- **Rule Coverage**: 100% of rows processed with cleaning rules

### Key Improvements by Column:
- **Item**: 1,213 missing values filled
- **Price Per Unit**: 609 missing values filled
- **Quantity**: 604 missing values filled
- **Total Spent**: 604 missing values filled
- **Discount Applied**: 4,199 missing values filled

## Development

### Adding New Dependencies

```bash
uv add package_name
```

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Formatting

```bash
uv run black src/
uv run flake8 src/
```

### Modifying Configuration

Edit `src/firstbase/config.py` to change default settings:

```python
@dataclass
class OpenAIConfig:
    model: str = "gpt-4"  # Change default model
    temperature: float = 0.1  # Change default temperature
```

```python
@dataclass
class DataConfig:
    """Data processing configuration."""

    sample_size: int = 20 # Change default sample size
    output_dir: str = "data/processed" # Change default output directory
    rules_filename: str = "cleaning_rules.json" # Change default rules filename
    cleaned_data_filename: str = "cleaned_data.csv" # Change default cleaned data filename
    metrics_filename: str = "cleaning_metrics.json" # Change default metrics filename
    data_description: str = """
    The Dirty Retail Store Sales dataset contains 12,575 rows of synthetic data representing sales transactions from a retail store.
    The dataset includes eight product categories with 25 items per category, each having static prices.
    It is designed to simulate real-world sales data, including intentional "dirtiness" such as missing or inconsistent values.
    This dataset is suitable for practicing data cleaning, exploratory data analysis (EDA), and feature engineering.
    """ # Change default data description
```

## Next Steps

1. **Alerts for Anomalies and Coveragen**:  Implements alerts for anomalies and data quality issues, integrates with Slack for real-time notifications
2. **API Development**: Create REST API endpoints for rule generation and application
3. **Track and Visualize Metrics**: Connects logs and alerts with Datadog to track, visualize and build comprehensive monitoring dashboards
4. **Containerization**: Docker containerization for easy deployment
5. **CI/CD**: Automated testing and deployment pipelines
6. **Experimentation**: Run more experiments to generate better rules.
