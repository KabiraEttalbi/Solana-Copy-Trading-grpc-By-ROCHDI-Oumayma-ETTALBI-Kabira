"""
Dataset Validation Script
Validates CSV files before training the ML model
"""

import pandas as pd
import sys
import os

REQUIRED_COLUMNS = [
    'volume',
    'liquidity',
    'holders',
    'tx_count',
    'price_change_1m',
    'price_change_5m',
    'volatility',
    'market_cap',
    'dev_activity',
    'label'
]

def validate_dataset(filepath):
    """Validate dataset structure and content"""
    
    print("=" * 70)
    print(f"Dataset Validation: {filepath}")
    print("=" * 70)
    print()
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found: {filepath}")
        return False
    
    # Load CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✓ File loaded successfully")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load CSV: {e}")
        return False
    
    print()
    print("Column Validation:")
    print("-" * 70)
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Found columns: {list(df.columns)}")
        return False
    else:
        print(f"✓ All required columns present")
    
    print()
    print("Data Quality Checks:")
    print("-" * 70)
    
    all_valid = True
    
    for col in REQUIRED_COLUMNS:
        print(f"\n{col}:")
        
        # Check for missing values
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  ⚠ Missing values: {missing} ({missing/len(df)*100:.1f}%)")
            all_valid = False
        else:
            print(f"  ✓ No missing values")
        
        # Check data type
        if col == 'label':
            if not all(df[col].isin([0, 1])):
                print(f"  ❌ Label must be 0 or 1")
                all_valid = False
            else:
                print(f"  ✓ Label values: 0 or 1")
            
            # Check label distribution
            label_counts = df[col].value_counts()
            print(f"  ✓ Label distribution:")
            for label, count in label_counts.items():
                pct = count / len(df) * 100
                print(f"    - {label}: {count} ({pct:.1f}%)")
        
        elif col == 'holders' or col == 'tx_count':
            if df[col].dtype not in ['int64', 'int32', 'int']:
                print(f"  ⚠ Should be integer, got {df[col].dtype}")
            else:
                print(f"  ✓ Correct data type (integer)")
            
            if (df[col] <= 0).any():
                print(f"  ⚠ Some values <= 0")
            else:
                print(f"  ✓ All values > 0")
        
        else:
            if df[col].dtype not in ['float64', 'float32', 'float']:
                print(f"  ⚠ Should be numeric, got {df[col].dtype}")
            else:
                print(f"  ✓ Correct data type (numeric)")
            
            # Check value ranges
            if col in ['volume', 'liquidity', 'market_cap']:
                if (df[col] <= 0).any():
                    print(f"  ⚠ Some values <= 0")
                else:
                    print(f"  ✓ All values > 0")
            
            elif col in ['volatility', 'dev_activity']:
                if (df[col] < 0).any() or (df[col] > 100).any():
                    print(f"  ⚠ Values should be 0-100, found min={df[col].min():.2f}, max={df[col].max():.2f}")
                else:
                    print(f"  ✓ Values in range 0-100")
            
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"  ✓ Range: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}")
    
    print()
    print("=" * 70)
    
    if all_valid and missing == 0:
        print("✓ Dataset validation PASSED!")
        print()
        print("Ready to train. Run:")
        print(f"  python ml/train.py {filepath}")
    else:
        print("⚠ Dataset validation completed with warnings.")
        print("  Review issues above before training.")
    
    print("=" * 70)
    
    return all_valid

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <path_to_csv>")
        print()
        print("Example: python validate_dataset.py ml/data/sample_trading_data.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = validate_dataset(filepath)
    sys.exit(0 if success else 1)
