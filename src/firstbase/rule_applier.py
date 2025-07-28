import json
import logging
from pathlib import Path
from typing import Any
from .cleaned_rule_generator import RuleType

import pandas as pd

logger = logging.getLogger(__name__)


class RuleApplier:
    """
    Reads a rules JSON file (exported by the rule‑generator LLM) and applies every
    rule to a pandas DataFrame.  Returns a *new* DataFrame plus an execution log.

    Supported rule_type values:
        • transformation   – regex replacement
        • anomaly_flag     – create bool flag column
        • imputation       – fill in missing values
        • format_string    – %-style numeric formatting    """

    def __init__(self, rules_path: str | Path):
        self.rules_path = Path(rules_path)
        self.rules: list[dict[str, Any]] = self._load_rules(self.rules_path)

    @staticmethod
    def _load_rules(path: Path | str) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload["rules"]

    def apply_rules(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Return a cleaned copy of *df* and a list of log entries."""
        df = df.copy(deep=True)
        log: list[dict[str, Any]] = []

        for rule in self.rules:
            rule_type = rule.get("rule_type", "transformation").lower()
            cols: list[str] = rule.get("columns", [])
            pattern: str | None = rule.get("pattern")
            replacement: str | None = rule.get("replacement")
            rule_id: str = rule.get("rule_id")
            description: str = rule.get("description", "")
            reasoning: str = rule.get("reasoning", "")
            flag_on_match: bool = rule.get("flag_on_match", False)

            if not cols:
                logger.warning("No columns specified for rule %s – skipping.", rule_id)
                continue

            for col in cols:
                if col not in df.columns:
                    logger.warning(
                        "Column '%s' missing in DataFrame – rule %s skipped.",
                        col,
                        rule_id,
                    )
                    continue

                if rule_type == RuleType.ANOMALY_FLAG:
                    self._apply_anomaly_flag(
                        df,
                        col,
                        pattern,
                        rule_id,
                        description,
                        reasoning,
                        flag_on_match,
                        log,
                    )

                elif rule_type == RuleType.IMPUTATION:
                    self._apply_imputation(
                        df,
                        col,
                        pattern,
                        replacement,
                        rule_id,
                        description,
                        reasoning,
                        log,
                    )

                elif rule_type == RuleType.FORMAT_STRING:
                    self._apply_format_string(
                        df,
                        col,
                        pattern,
                        replacement,
                        rule_id,
                        description,
                        reasoning,
                        log,
                    )

                elif rule_type == RuleType.TRANSFORMATION:
                    self._apply_transformation(
                        df,
                        col,
                        pattern,
                        replacement,
                        rule_id,
                        description,
                        reasoning,
                        log,
                    )

                else:
                    logger.warning(
                        "Unknown rule type '%s' for rule %s – skipping.", rule_type, rule_id
                    )

        return df, log

    def _apply_anomaly_flag(
        self,
        df: pd.DataFrame,
        col: str,
        pattern: str,
        rule_id: str,
        description: str,
        reasoning: str,
        flag_on_match: bool,
        log: list,
    ) -> None:
        """Create <col>_flag_<rule_id> with True for invalid rows."""
        flag_col = f"{col}_flag_{rule_id}"

        # special arithmetic sanity‑check
        if rule_id == "flag_total_spent_mismatch":
            tol = 0.01  # 1 cent
            expected = (df["Price Per Unit"] * df["Quantity"]).round(2)
            mask_invalid = (df[col] - expected).abs().gt(tol)

        else:
            match_mask = df[col].astype(str).str.match(pattern, na=False)
            mask_invalid = match_mask if flag_on_match else ~match_mask

        df[flag_col] = mask_invalid

        log.append(
            {
                "rule_id": rule_id,
                "type": "anomaly_flag",
                "column": col,
                "flagged_count": int(mask_invalid.sum()),
                "description": description,
                "reasoning": reasoning,
            }
        )

    def _apply_imputation(
        self,
        df: pd.DataFrame,
        col: str,
        pattern: str,
        replacement: str,
        rule_id: str,
        description: str,
        reasoning: str,
        log: list,
    ) -> None:
        mask_missing = df[col].astype(str).str.match(pattern, na=False)

        if replacement == "[Price Per Unit] * [Quantity]":
            valid = (
                df["Price Per Unit"].notna() & df["Quantity"].notna() & mask_missing
            )
            df.loc[valid, col] = df.loc[valid, "Price Per Unit"] * df.loc[
                valid, "Quantity"
            ]

        elif replacement == "<MEDIAN_FROM_GROUP_OR_GLOBAL>":
            if "Item" in df.columns:
                medians = df.groupby("Item")[col].median()
                df.loc[mask_missing, col] = df.loc[mask_missing, "Item"].map(medians)
            df[col].fillna(df[col].median(), inplace=True)

        else:
            df.loc[mask_missing, col] = replacement

        log.append(
            {
                "rule_id": rule_id,
                "type": "imputation",
                "column": col,
                "imputed_count": int(mask_missing.sum()),
                "description": description,
                "reasoning": reasoning,
            }
        )

    def _apply_format_string(
        self,
        df: pd.DataFrame,
        col: str,
        pattern: str | None,
        replacement: str,
        rule_id: str,
        description: str,
        reasoning: str,
        log: list,
    ) -> None:
        try:
            if "%" in replacement:
                # numeric %-formatting (keep as float)
                df[col] = df[col].apply(
                    lambda x: float(replacement % float(x)) if pd.notna(x) else x
                )
            elif pattern:
                df[col] = df[col].astype(str).str.replace(
                    pattern, replacement, regex=True
                )
            else:
                df[col] = df[col].apply(lambda x: replacement % float(x))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Format‑string rule %s failed on column %s: %s", rule_id, col, exc
            )

        log.append(
            {
                "rule_id": rule_id,
                "type": "format_string",
                "column": col,
                "description": description,
                "reasoning": reasoning,
            }
        )

    def _apply_transformation(
        self,
        df: pd.DataFrame,
        col: str,
        pattern: str,
        replacement: str,
        rule_id: str,
        description: str,
        reasoning: str,
        log: list,
    ) -> None:
        df[col] = df[col].astype(str).str.replace(pattern, replacement, regex=True)

        log.append(
            {
                "rule_id": rule_id,
                "type": "transformation",
                "column": col,
                "description": description,
                "reasoning": reasoning,
            }
        )
