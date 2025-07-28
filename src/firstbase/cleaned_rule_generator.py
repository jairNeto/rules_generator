from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
from enum import StrEnum
from .config import DEFAULT_CONFIG


class RuleType(StrEnum):
    """Enumeration of rule types for data cleaning."""

    ANOMALY_FLAG = "anomaly_flag"
    IMPUTATION = "imputation"
    TRANSFORMATION = "transformation"
    FORMAT_STRING = "format_string"


class Rule(BaseModel):
    """
    One data-cleaning transformation rule.
    """

    reasoning: str = Field(
        ...,
        description=(
            "Why you are generating this rules and why this rule is important."
            "Example: 'Phone numbers are fields that are often incomplete or incorrect, so we need to normalize them.'"
        ),
    )
    rule_id: str = Field(
        ...,
        description=(
            "A stable, snake_case identifier for the rule. "
            "It must be unique across all rules, e.g. 'normalize_phone'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "A short, imperative sentence (≤ 120 chars) explaining what the rule does. "
            "Example: 'Normalize phone numbers to +1-XXX-XXX-XXXX format'."
        ),
    )
    confidence: float = Field(
        ...,
        description=(
            "A float between 0 and 1 representing the LLM's confidence for rule application. "
            "Higher means more certain. "
            "Example: 0.94"
        ),
    )
    pattern: str = Field(
        ...,
        description=(
            "A raw regular-expression (Python-style) used to detect values that require "
            "transformation. Backslashes must be escaped in JSON. "
            "Example: r'\\d{3}[. -]?\\d{3}[. -]?\\d{4}'"
        ),
    )
    replacement: str = Field(
        ...,
        description=(
            "Replacement string applied with back-references (\\1, \\2, …) that defines "
            "the canonical form once the pattern matches. "
            "Example: +1-\\1-\\2-\\3"
        ),
    )
    rule_type: RuleType = Field(
        ...,
        description=(
            "The type of rule: 'anomaly_flag' for flagging suspicious data, "
            "'imputation' for filling missing values, 'transformation' for general changes, "
            "or 'format_string' for special formatting like rounding."
        ),
    )
    columns: list[str] = Field(
        ...,
        description=(
            "List of column names to which this rule should be applied. "
            "Example: ['phone_number', 'mobile_number']"
        ),
    )


class RulesResponse(BaseModel):
    """
    Top-level container returned by the LLM.
    """

    rules: list[Rule] = Field(
        ..., description="An ordered list of data-cleaning rules."
    )


class CleanedRuleGenerator:

    SYSTEM_PROMPT = """
    # Role

    You are a data cleaning expert. You are given a sample of a messy dataset and you need to generate a list of data cleaning rules.

    # Instructions
    
    Ensure you create a *comprehensive* set of rules that covers both data-cleaning transformations *and* possible anomaly-detection checks.
    Your goal is to maximize recall, so include every rule that could plausibly be needed—even if the confidence is moderate—so long as it improves overall coverage.
    Use the provided sample records only as illustrative examples; do *not* assume the full dataset will always follow those same patterns.
    Generalize cautiously, assign an appropriate confidence score to each rule, and clearly flag any assumptions you make.

    The data cleaning rules should be:
    - Human-readable
    - Generalizable
    - Derived from learned patterns rather than hardcoded heuristics
    - Explainable
    - Production-ready

    These rules may include:
    - Duplicate detection and merging (using fuzzy matching or embeddings)
    - Phone number normalization
    - Name and address formatting (e.g., title casing)
    - Splitting or standardizing multi-value fields
    - Replacing or flagging suspicious, incomplete, or outlier values

    These rule MUST include:
    - At least one rule for anomaly detection (e.g., outliers, unexpected formats, etc.)

    # Input Format
    
    You will be given a sample of a messy dataset in a tabular format, the description of the dataset and the types of the columns.

    # Output Format
    The output should be a list of data cleaning rules. Each rule should be a dictionary with the following keys:
    - rule_id: A stable, snake_case identifier for the rule.
    - description: A short, imperative sentence (≤ 120 chars) explaining what the rule does.
    - confidence: A float between 0 and 1 representing the LLM's confidence for rule application. Higher means more certain.
    - pattern: A raw regular-expression (Python-style) used to detect values that require transformation. Backslashes must be escaped in JSON.
    - replacement: Replacement string applied with back-references (\\1, \\2, …) that defines the canonical form once the pattern matches.
    - rule_type: One of "anomaly_flag", "imputation", "transformation", or "format_string":
        - "anomaly_flag": For flagging suspicious or outlier data (creates flag columns)
        - "imputation": For filling missing values with calculated or default values
        - "transformation": For general data transformations and normalizations
        - "format_string": For special formatting like rounding numbers (uses % format strings)
    - columns: List of column names to which this rule should be applied.

    # Example

    ```
    [
        {
            "rule_id": "normalize_phone",
            "description": "Normalize phone numbers to +1-XXX-XXX-XXXX format",
            "confidence": 0.94,
            "pattern": "(\\d{3})[. -]?(\\d{3})[. -]?(\\d{4})",
            "replacement": "+1-\1-\2-\3",
            "rule_type": "transformation",
            "columns": ["phone_number"]
        }
    ]
    ```
    """

    def __init__(
        self, api_key: str, model: str | None = None, temperature: float | None = None
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_CONFIG.openai.model
        self.temperature = temperature or DEFAULT_CONFIG.openai.temperature

    @staticmethod
    def _format_input(data: pd.DataFrame, data_description: str) -> str:
        """
        Format the input for the LLM, including the dataset description, the columns description and the dataset sample.

        Args:
            data (pd.DataFrame): The dataset to be cleaned.
            data_description (str): The description of the dataset.

        Returns:
            str: The formatted input for the LLM.
        """
        user_prompt_format = (
            "# Dataset description\n\n"
            f"{data_description}\n\n"
            "# Dataset Columns Description\n\n"
            f"{data.dtypes.to_markdown()}\n\n"
            "# Dataset sample\n\n"
            f"{data.to_markdown()}"
        )

        return user_prompt_format

    def generate_cleaned_rules(
        self, data: pd.DataFrame, data_description: str
    ) -> RulesResponse:
        user_prompt = self._format_input(data, data_description)
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            text_format=RulesResponse,
        )

        return response.output_parsed
