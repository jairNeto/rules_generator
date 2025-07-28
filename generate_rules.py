#!/usr/bin/env python3
"""Script to generate cleaning rules using the CleanedRuleGenerator."""

import sys
import argparse
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from firstbase.data_loader import DataLoader
from firstbase.cleaned_rule_generator import CleanedRuleGenerator
from firstbase.data_writer import DataWriter
from firstbase.config import DEFAULT_CONFIG


def main():
    """Main function to generate cleaning rules."""
    # Load environment variables (only for API key)
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate cleaning rules for a CSV file")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY in .env)")
    parser.add_argument("--model", default=DEFAULT_CONFIG.openai.model, help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.openai.temperature, help="Temperature for generation")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_CONFIG.data.sample_size, help="Number of rows to sample for rule generation")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.data.output_dir, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OpenAI API key is required. Set it via --api-key or OPENAI_API_KEY in .env file")
        sys.exit(1)
    
    # Validate CSV file exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("🚀 Starting rule generation process...")
    print(f"🤖 Using model: {args.model}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"📋 Sample size: {args.sample_size}")
    print(f"📁 Output directory: {args.output_dir}")
    
    try:
        # Load data
        print("📊 Loading data...")
        loader = DataLoader(str(csv_path))
        df = loader.load_data()
        
        # Sample data for rule generation (to avoid token limits)
        if len(df) > args.sample_size:
            print(f"📋 Sampling {args.sample_size} rows for rule generation...")
            sample_df = df.sample(n=args.sample_size, random_state=42)
        else:
            sample_df = df
        
        # Use data description from config
        data_description = DEFAULT_CONFIG.data.data_description
        
        # Initialize rule generator
        print("🤖 Initializing rule generator...")
        rule_generator = CleanedRuleGenerator(
            api_key=api_key,
            model=args.model,
            temperature=args.temperature
        )
        
        # Generate rules
        print("🔧 Generating cleaning rules...")
        rules_response = rule_generator.generate_cleaned_rules(sample_df, data_description)
        
        # Convert rules to dictionary format
        rules_dict = [rule.model_dump() for rule in rules_response.rules]
        
        print(f"✅ Generated {len(rules_dict)} cleaning rules")
        
        # Write rules to JSON
        print("💾 Writing rules to JSON...")
        writer = DataWriter(args.output_dir)
        rules_path = writer.write_rules_to_json(rules_dict)
        
        # Print summary
        print("\n📋 Generated Rules Summary:")
        for i, rule in enumerate(rules_dict, 1):
            print(f"  {i}. {rule['rule_id']}: {rule['description']} (confidence: {rule['confidence']:.2f})")
        
        print(f"\n🎉 Success! Rules written to: {rules_path}")
        print(f"📁 Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during rule generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 