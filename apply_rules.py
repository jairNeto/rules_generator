#!/usr/bin/env python3
"""Script to apply cleaning rules to a CSV file using the RuleApplier."""

import sys
import argparse
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from firstbase.data_loader import DataLoader
from firstbase.rule_applier import RuleApplier
from firstbase.data_writer import DataWriter
from firstbase.config import DEFAULT_CONFIG


def main():
    """Main function to apply cleaning rules to a CSV file."""
    parser = argparse.ArgumentParser(description="Apply cleaning rules to a CSV file")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("--rules-path", default="data/processed/cleaning_rules.json", 
                       help="Path to the JSON rules file")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.data.output_dir, 
                       help="Output directory for results")
    parser.add_argument("--output-filename", default="cleaned_data.csv", 
                       help="Output filename for cleaned data")
    parser.add_argument("--log-filename", default="cleaning_log.json", 
                       help="Output filename for cleaning log")
    parser.add_argument("--show-summary", action="store_true", 
                       help="Show a summary of applied rules")
    
    args = parser.parse_args()
    
    # Validate input files exist
    csv_path = Path(args.csv_path)
    rules_path = Path(args.rules_path)
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    if not rules_path.exists():
        print(f"❌ Error: Rules file not found: {rules_path}")
        sys.exit(1)
    
    print("🚀 Starting rule application process...")
    print(f"📊 Input CSV: {csv_path}")
    print(f"🔧 Rules file: {rules_path}")
    print(f"📁 Output directory: {args.output_dir}")
    
    try:
        # Load data
        print("📊 Loading data...")
        loader = DataLoader(str(csv_path))
        df = loader.load_data()
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Load and apply rules
        print("🔧 Loading and applying rules...")
        applier = RuleApplier(str(rules_path))
        cleaned_df, log = applier.apply_rules(df)
        
        print(f"✅ Applied {len(log)} rules")
        
        # Write results
        print("💾 Writing results...")
        writer = DataWriter(args.output_dir)
        
        # Write cleaned data
        cleaned_data_path = writer.write_cleaned_data_to_csv(
            cleaned_df, 
            filename=args.output_filename
        )
        
        # Write cleaning log
        log_path = writer.write_metrics_to_json(
            {
                "total_rules_applied": len(log),
                "rules_applied": log,
                "original_shape": df.shape,
                "cleaned_shape": cleaned_df.shape,
                "columns_added": list(set(cleaned_df.columns) - set(df.columns))
            },
            filename=args.log_filename
        )
        
        # Show summary if requested
        if args.show_summary:
            print("\n📋 Rule Application Summary:")
            print("=" * 50)
            
            for i, rule_log in enumerate(log, 1):
                rule_type = rule_log.get("type", "unknown")
                column = rule_log.get("column", "unknown")
                description = rule_log.get("description", "No description")
                
                if rule_type == "anomaly_flag":
                    flagged_count = rule_log.get("flagged_count", 0)
                    print(f"  {i}. 🚩 {rule_log['rule_id']}: {description}")
                    print(f"     Column: {column}, Flagged: {flagged_count} rows")
                elif rule_type == "imputation":
                    imputed_count = rule_log.get("imputed_count", 0)
                    print(f"  {i}. 🔧 {rule_log['rule_id']}: {description}")
                    print(f"     Column: {column}, Imputed: {imputed_count} rows")
                else:
                    print(f"  {i}. ✏️  {rule_log['rule_id']}: {description}")
                    print(f"     Column: {column}")
                print()
        
        # Show basic statistics
        print("\n📊 Data Statistics:")
        print(f"  Original rows: {len(df)}")
        print(f"  Cleaned rows: {len(cleaned_df)}")
        print(f"  Original columns: {len(df.columns)}")
        print(f"  Cleaned columns: {len(cleaned_df.columns)}")
        
        # Show new columns (flags)
        new_columns = set(cleaned_df.columns) - set(df.columns)
        if new_columns:
            print(f"  New flag columns: {len(new_columns)}")
            for col in sorted(new_columns):
                flag_count = cleaned_df[col].sum()
                print(f"    - {col}: {flag_count} flagged rows")
        
        print(f"\n🎉 Success! Results written to:")
        print(f"  📄 Cleaned data: {cleaned_data_path}")
        print(f"  📋 Cleaning log: {log_path}")
        
    except Exception as e:
        print(f"❌ Error during rule application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 