"""Unit tests for the RuleApplier class."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Ensure `firstbase` package is importable when tests are run from repo root
sys.path.append(str(Path(__file__).parent.parent / "src"))

from firstbase.rule_applier import RuleApplier


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Transaction ID": ["TXN_001", "TXN_002", "TXN_003"],
            "Category": ["food", "BEVERAGES", "Food"],
            "Item": ["Item_1_FOOD", "Item_2_BEV", None],
            "Price Per Unit": [10.5, 20.0, 15.753],
            "Quantity": [2, 1, 3],
            "Total Spent": [21.0, 20.0, None],
            "Payment Method": ["Credit Card", "Digital Wallet", "Cash"],
            "Location": ["Online", "In-store", "Online"],
            "Transaction Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Discount Applied": [True, False, None],
        }
    )


@pytest.fixture
def sample_rules() -> dict[str, Any]:
    return {
        "metadata": {
            "total_rules": 5,
            "generated_at": "2024-01-01T00:00:00",
            "description": "Test rules",
        },
        "rules": [
            {
                "rule_id": "normalize_category",
                "description": "Convert Category to title case",
                "confidence": 0.95,
                "pattern": r"^(.*)$",
                "replacement": r"\1",
                "columns": ["Category"],
                "rule_type": "transformation",
                "reasoning": "Standardize category names",
            },
            {
                "rule_id": "flag_missing_item",
                "description": "Flag missing Item values",
                "confidence": 0.9,
                "pattern": r"^(nan|NULL|NaN|None|\s*)$",
                "replacement": "[MISSING_ITEM]",
                "columns": ["Item"],
                "rule_type": "anomaly_flag",
                "flag_on_match": True,  # ← flag when pattern MATCHES
                "reasoning": "Flag missing items for review",
            },
            {
                "rule_id": "impute_total_spent",
                "description": "Impute Total Spent as Price * Quantity",
                "confidence": 0.85,
                "pattern": r"^(nan|NULL|NaN|None|\s*)$",
                "replacement": "[Price Per Unit] * [Quantity]",
                "columns": ["Total Spent"],
                "rule_type": "imputation",
                "reasoning": "Calculate missing totals",
            },
            {
                "rule_id": "round_price",
                "description": "Round Price Per Unit to 2 decimals",
                "confidence": 0.98,
                "pattern": r"^(\d+\.\d{3,})$",
                "replacement": "%.2f",
                "columns": ["Price Per Unit"],
                "rule_type": "format_string",
                "reasoning": "Standardize price formatting",
            },
            {
                "rule_id": "impute_discount",
                "description": "Impute missing Discount Applied as False",
                "confidence": 0.8,
                "pattern": r"^(nan|NULL|NaN|None|\s*)$",
                "replacement": "False",
                "columns": ["Discount Applied"],
                "rule_type": "imputation",
                "reasoning": "Assume no discount if missing",
            },
        ],
    }


@pytest.fixture
def rules_file(sample_rules: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(sample_rules), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Parametrised functional tests (one row per rule)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rule_id,expectation",
    [
        (
            "normalize_category",
            lambda df: df["Category"].tolist() == ["food", "BEVERAGES", "Food"],
        ),
        (
            "flag_missing_item",
            lambda df: df["Item_flag_flag_missing_item"].tolist()
            == [False, False, True],
        ),
        (
            "impute_total_spent",
            lambda df: df["Total Spent"].iloc[2] == 15.753 * 3,
        ),
        (
            "round_price",
            lambda df: df["Price Per Unit"].iloc[2] == 15.75,
        ),
        (
            "impute_discount",
            lambda df: df["Discount Applied"].iloc[2] == "False",
        ),
    ],
)
def test_each_rule_works(
    sample_df: pd.DataFrame,
    rules_file: Path,
    rule_id: str,
    expectation,
) -> None:
    applier = RuleApplier(rules_file)
    cleaned, logs = applier.apply_rules(sample_df)
    assert expectation(cleaned)
    assert any(entry["rule_id"] == rule_id for entry in logs)


# --------------------------------------------------------------------------- #
# Edge‑case / regression tests
# --------------------------------------------------------------------------- #
def test_missing_rules_file() -> None:
    with pytest.raises(FileNotFoundError):
        RuleApplier("nonexistent.json")


def test_invalid_regex_pattern(
    sample_df: pd.DataFrame, rules_file: Path, sample_rules: dict[str, Any]
) -> None:
    sample_rules["rules"][0]["pattern"] = "invalid[regex"
    rules_file.write_text(json.dumps(sample_rules), encoding="utf-8")
    applier = RuleApplier(rules_file)
    with pytest.raises(Exception):
        applier.apply_rules(sample_df)


def test_empty_dataframe(rules_file: Path) -> None:
    applier = RuleApplier(rules_file)
    cleaned, log = applier.apply_rules(pd.DataFrame())
    assert cleaned.empty and not log
