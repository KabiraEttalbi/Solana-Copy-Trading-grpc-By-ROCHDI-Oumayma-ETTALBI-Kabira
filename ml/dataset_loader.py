import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

class KaggleDatasetLoader:
    """
    Loads and preprocesses trading data from Kaggle datasets.
    Prepares data for model training and validation.
    """
    
    def __init__(self, dataset_path="./ml/data"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.features = None
        self.labels = None
    
    def load_csv(self, file_path):
        """Load CSV file from Kaggle dataset"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"Loaded dataset with shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    
    def preprocess(self):
        """Clean and prepare data for model training"""
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_csv() first.")
        
        # Handle missing values
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Feature engineering
        self._engineer_features()
        
        return self.df
    
    def _engineer_features(self):
        """Create trading-specific features from raw data"""
        # Calculate technical indicators if price data exists
        if 'close' in self.df.columns:
            self.df['price_change_1m'] = self.df['close'].pct_change(periods=1).fillna(0)
            self.df['price_change_5m'] = self.df['close'].pct_change(periods=5).fillna(0)
            self.df['volatility'] = self.df['close'].rolling(window=10).std().fillna(0)
        
        # Normalize large numbers
        if 'volume' in self.df.columns:
            self.df['volume'] = np.log1p(self.df['volume'])
        
        if 'market_cap' in self.df.columns:
            self.df['market_cap'] = np.log1p(self.df['market_cap'])
    
    def create_training_set(self, target_column='profit', test_size=0.2):
        """
        Create training and validation sets.
        
        Args:
            target_column: Column name that indicates if trade was profitable
            test_size: Proportion of data for testing
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        # Identify feature columns (exclude identifiers and target)
        exclude_cols = [target_column, 'id', 'timestamp', 'date', 'token_id']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].values
        y = self.df[target_column].values if target_column in self.df.columns else np.random.randint(0, 2, len(self.df))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.features = X_train
        self.labels = y_train
        
        return X_train, X_test, y_train, y_test
    
    def get_sample_features(self, token_data):
        """
        Convert real-time token data to feature vector.
        
        Args:
            token_data: Dict with token market data
        
        Returns:
            Feature vector for prediction
        """
        feature_order = [
            'volume', 'liquidity', 'holder_count', 'tx_count',
            'price_change_1m', 'price_change_5m', 'volatility',
            'market_cap', 'created_timestamp', 'dev_activity'
        ]
        
        features = []
        for feature in feature_order:
            value = token_data.get(feature, 0)
            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def save_processed_data(self, output_path="./ml/data/processed"):
        """Save preprocessed data for faster loading"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        if self.df is not None:
            self.df.to_csv(f"{output_path}/processed_data.csv", index=False)
            print(f"Saved processed data to {output_path}")
        
        if self.features is not None:
            np.save(f"{output_path}/features.npy", self.features)
            np.save(f"{output_path}/labels.npy", self.labels)
            print("Saved features and labels")
    
    def load_processed_data(self, data_path="./ml/data/processed"):
        """Load previously processed data"""
        try:
            self.features = np.load(f"{data_path}/features.npy")
            self.labels = np.load(f"{data_path}/labels.npy")
            self.df = pd.read_csv(f"{data_path}/processed_data.csv")
            print("Loaded processed data successfully")
            return self.df
        except FileNotFoundError:
            print("Processed data not found. Run preprocess() first.")
            return None
    
    def get_statistics(self):
        """Return dataset statistics"""
        if self.df is None:
            return None
        
        return {
            "total_samples": len(self.df),
            "features": len(self.df.columns),
            "missing_values": int(self.df.isnull().sum().sum()),
            "numeric_columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "shape": self.df.shape
        }


def load_and_prepare_dataset(dataset_path):
    """
    Convenience function to load and prepare dataset.
    
    Args:
        dataset_path: Path to CSV file from Kaggle
    
    Returns:
        Prepared DataLoader instance
    """
    loader = KaggleDatasetLoader()
    loader.load_csv(dataset_path)
    loader.preprocess()
    X_train, X_test, y_train, y_test = loader.create_training_set()
    
    return loader, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset_loader.py <path_to_kaggle_csv>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    loader, (X_train, X_test, y_train, y_test) = load_and_prepare_dataset(dataset_path)
    
    print("\nDataset Statistics:")
    print(json.dumps(loader.get_statistics(), indent=2))
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training labels: {y_train.shape}")
