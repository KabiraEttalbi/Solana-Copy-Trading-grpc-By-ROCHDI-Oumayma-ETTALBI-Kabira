"""
Sample Dataset Generator for Solana Trading Bot
Generates synthetic trading data for model training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_dataset(num_samples=1000, output_file='ml/data/sample_trading_data.csv'):
    """
    Generate a synthetic dataset for training the ML model.
    
    Parameters:
    - num_samples: Number of trading samples to generate
    - output_file: Path to save the CSV file
    
    Features:
    - volume: Trading volume
    - liquidity: Pool liquidity in SOL
    - holders: Number of token holders
    - tx_count: Transaction count
    - price_change_1m: 1-minute price change %
    - price_change_5m: 5-minute price change %
    - volatility: Price volatility score
    - market_cap: Market capitalization
    - dev_activity: Developer activity score
    - label: 1 (profitable trade) or 0 (not profitable)
    """
    
    np.random.seed(42)
    
    data = {
        'volume': np.random.lognormal(10, 2, num_samples),
        'liquidity': np.random.lognormal(5, 1.5, num_samples),
        'holders': np.random.randint(10, 10000, num_samples),
        'tx_count': np.random.randint(5, 500, num_samples),
        'price_change_1m': np.random.normal(0, 5, num_samples),
        'price_change_5m': np.random.normal(0, 8, num_samples),
        'volatility': np.random.uniform(0, 100, num_samples),
        'market_cap': np.random.lognormal(15, 2, num_samples),
        'dev_activity': np.random.uniform(0, 100, num_samples),
    }
    
    # Create realistic labels based on features
    # Profitable trades tend to have: high volume, good liquidity, positive price change
    df = pd.DataFrame(data)
    
    labels = []
    for idx, row in df.iterrows():
        score = 0
        
        # Volume score
        if row['volume'] > df['volume'].quantile(0.75):
            score += 1
        
        # Liquidity score
        if row['liquidity'] > df['liquidity'].quantile(0.70):
            score += 1
        
        # Holders score
        if row['holders'] > 100:
            score += 1
        
        # Price change score (positive is good)
        if row['price_change_1m'] > 0:
            score += 1
        if row['price_change_5m'] > 0:
            score += 0.5
        
        # Volatility score (moderate volatility is good, extreme is bad)
        if 20 < row['volatility'] < 80:
            score += 0.5
        
        # Dev activity score
        if row['dev_activity'] > 30:
            score += 1
        
        # Add some randomness
        score += np.random.normal(0, 0.5)
        
        # Label: 1 if score >= 3, else 0
        labels.append(1 if score >= 3 else 0)
    
    df['label'] = labels
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Sample dataset generated: {output_file}")
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Profitable trades (label=1): {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"✓ Non-profitable trades (label=0): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    
    return df

def create_dataset_template():
    """Create a template showing the expected CSV format"""
    
    template = pd.DataFrame({
        'volume': [1000000, 500000, 2000000],
        'liquidity': [50000, 25000, 100000],
        'holders': [500, 250, 1000],
        'tx_count': [100, 50, 200],
        'price_change_1m': [2.5, -1.0, 5.0],
        'price_change_5m': [3.0, 0.5, 7.5],
        'volatility': [45.0, 60.0, 35.0],
        'market_cap': [1000000000, 500000000, 2000000000],
        'dev_activity': [75.0, 40.0, 85.0],
        'label': [1, 0, 1]
    })
    
    os.makedirs('ml/data', exist_ok=True)
    template.to_csv('ml/data/dataset_template.csv', index=False)
    print("✓ Dataset template created: ml/data/dataset_template.csv")
    return template

if __name__ == '__main__':
    print("=" * 60)
    print("Sample Dataset Generator for Solana Trading Bot")
    print("=" * 60)
    print()
    
    # Generate sample dataset
    print("Generating sample dataset...")
    df = generate_sample_dataset(num_samples=1000)
    print()
    
    # Create template
    print("Creating dataset template...")
    create_dataset_template()
    print()
    
    # Show statistics
    print("Dataset Statistics:")
    print(df.describe())
    print()
    print("=" * 60)
    print("Next steps:")
    print("1. Use sample_trading_data.csv for testing")
    print("2. Replace with your Kaggle dataset for production")
    print("3. Run: python ml/train.py ml/data/sample_trading_data.csv")
    print("=" * 60)
