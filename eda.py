#!/usr/bin/env python3
"""Exploratory Data Analysis (EDA) script for the Firstbase AI Challenge."""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from firstbase.data_loader import DataLoader

def print_separator(title: str):
    """Print a formatted separator with title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    """Main EDA function."""
    print("🚀 Starting Exploratory Data Analysis for Firstbase AI Challenge")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load data
    print_separator("LOADING DATA")
    try:
        df = loader.load_data()
        print("✅ Data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Basic data information
    print_separator("BASIC DATA INFORMATION")
    info = loader.get_data_info()
    
    print(f"📊 Dataset Shape: {info['shape']}")
    print(f"📋 Columns: {', '.join(info['columns'])}")
    print(f"💾 Memory Usage: {info['memory_usage'] / 1024:.2f} KB")
    print(f"🔄 Duplicate Rows: {info['duplicate_rows']}")
    
    # Data types
    print_separator("DATA TYPES")
    for col, dtype in info['dtypes'].items():
        print(f"  {col}: {dtype}")
    
    # Missing values analysis
    print_separator("MISSING VALUES ANALYSIS")
    null_counts = info['null_counts']
    total_rows = info['shape'][0]
    
    print("Missing values per column:")
    for col, count in null_counts.items():
        if count > 0:
            percentage = (count / total_rows) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")
        else:
            print(f"  {col}: 0 (0.00%)")
    
    # Sample data
    print_separator("SAMPLE DATA (First 5 rows)")
    sample_data = loader.get_sample_data(5)
    print(sample_data.to_string(index=False))
    
    # Column-wise analysis
    print_separator("COLUMN-WISE ANALYSIS")
    
    for column in df.columns:
        print(f"\n📈 {column}:")
        try:
            stats = loader.get_column_stats(column)
            
            print(f"  Data Type: {stats['dtype']}")
            print(f"  Null Count: {stats['null_count']} ({stats['null_percentage']:.2f}%)")
            print(f"  Unique Values: {stats['unique_count']} ({stats['unique_percentage']:.2f}%)")
            
            # Additional stats for numeric columns
            if 'mean' in stats:
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Median: {stats['median']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
            
            # Show unique values for categorical columns (if not too many)
            if stats['unique_count'] <= 20 and stats['unique_count'] > 0:
                unique_vals = df[column].dropna().unique()
                print(f"  Unique Values: {unique_vals}")
                
        except Exception as e:
            print(f"  Error analyzing column: {e}")
    
    # Data quality issues
    print_separator("POTENTIAL DATA QUALITY ISSUES")
    
    # Check for data inconsistencies
    issues = []
    
    # 1. Check if Total Spent = Price Per Unit × Quantity
    if 'Price Per Unit' in df.columns and 'Quantity' in df.columns and 'Total Spent' in df.columns:
        mask = df['Price Per Unit'].notna() & df['Quantity'].notna() & df['Total Spent'].notna()
        if mask.any():
            calculated_total = df.loc[mask, 'Price Per Unit'] * df.loc[mask, 'Quantity']
            mismatches = abs(calculated_total - df.loc[mask, 'Total Spent']) > 0.01
            if mismatches.any():
                issues.append(f"Price calculation mismatch: {mismatches.sum()} rows")
    
    # 2. Check for suspicious values
    if 'Price Per Unit' in df.columns:
        zero_prices = (df['Price Per Unit'] == 0).sum()
        if zero_prices > 0:
            issues.append(f"Zero prices: {zero_prices} rows")
    
    if 'Quantity' in df.columns:
        zero_quantities = (df['Quantity'] == 0).sum()
        if zero_quantities > 0:
            issues.append(f"Zero quantities: {zero_quantities} rows")
    
    # 3. Check for future dates
    if 'Transaction Date' in df.columns:
        try:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
            future_dates = (df['Transaction Date'] > pd.Timestamp.now()).sum()
            if future_dates > 0:
                issues.append(f"Future dates: {future_dates} rows")
        except:
            pass
    
    if issues:
        print("⚠️  Potential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No obvious data quality issues detected")
    
    # Summary statistics
    print_separator("SUMMARY STATISTICS")
    print(df.describe(include='all'))
    
    print("\n🎉 EDA Complete!")
    print("\nNext steps:")
    print("1. Review the data quality issues identified")
    print("2. Plan data cleaning rules based on the analysis")
    print("3. Implement rule generation logic")

if __name__ == "__main__":
    main()
