#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.model import TradePredictionModel
from ml.dataset_loader import load_and_prepare_dataset

def train_model(dataset_path, output_path="./ml/models", epochs=50, batch_size=32):
    """
    Train the trade prediction model on Kaggle dataset.
    
    Args:
        dataset_path: Path to the Kaggle CSV dataset
        output_path: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training metrics
    """
    
    print("=" * 60)
    print("Solana Trade Prediction Model Training")
    print("=" * 60)
    
    # Check dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("\nTo download a Kaggle dataset:")
        print("1. Go to https://www.kaggle.com/datasets")
        print("2. Find a suitable trading/crypto dataset")
        print("3. Download the CSV file")
        print("4. Place it in ./ml/data/")
        sys.exit(1)
    
    print(f"\n📊 Loading dataset from: {dataset_path}")
    
    try:
        # Load and prepare dataset
        loader, (X_train, X_test, y_train, y_test) = load_and_prepare_dataset(dataset_path)
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Features per sample: {X_train.shape[1] if len(X_train.shape) > 1 else 1}")
        
        # Print dataset statistics
        stats = loader.get_statistics()
        print(f"\n📈 Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Initialize model
        print(f"\n🤖 Building neural network model...")
        model = TradePredictionModel()
        
        # Train model
        print(f"\n🚀 Training model for {epochs} epochs with batch size {batch_size}...")
        print("-" * 60)
        
        history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        print("-" * 60)
        print(f"✓ Training completed!")
        
        # Prepare test data with proper reshaping
        print(f"\n📊 Evaluating on test set...")
        X_test_processed = model.preprocess_data(X_test, fit_scaler=False)
        
        # Align labels with processed sequences (sequences are created with sliding window)
        # Each sequence is 10 timesteps, so we have len(X_test) - 10 + 1 sequences
        num_sequences = max(1, len(X_test) - 10 + 1)
        y_test_aligned = y_test[-num_sequences:] if len(y_test) >= num_sequences else y_test
        
        # If not enough labels, pad them
        if len(y_test_aligned) < len(X_test_processed):
            padding_size = len(X_test_processed) - len(y_test_aligned)
            y_test_aligned = np.concatenate([y_test_aligned, y_test_aligned[-padding_size:]])
        
        # Evaluate on test set
        test_loss, test_acc, test_auc = model.model.evaluate(
            X_test_processed, y_test_aligned, verbose=0
        )
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        
        # Save model
        Path(output_path).mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Saving model to: {output_path}")
        model.save_model(output_path)
        
        # Save training metrics
        metrics = {
            "test_accuracy": float(test_acc),
            "test_auc": float(test_auc),
            "test_loss": float(test_loss),
            "epochs": epochs,
            "batch_size": batch_size,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "dataset_path": dataset_path,
            "timestamp": __import__('time').time()
        }
        
        with open(f"{output_path}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Model and metrics saved successfully!")
        
        print(f"\n{'=' * 60}")
        print("✅ Training Complete!")
        print(f"{'=' * 60}")
        print(f"\nModel ready for predictions.")
        print(f"Model location: {output_path}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_model(model_path, test_data_path):
    """Evaluate existing model on new data"""
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    try:
        model = TradePredictionModel(model_path=model_path)
        
        # Load test data
        loader = load_and_prepare_dataset(test_data_path)
        X_train, X_test, y_train, y_test = loader[1]
        
        # Evaluate
        test_loss, test_acc, test_auc = model.model.evaluate(X_test, y_test, verbose=1)
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Loss: {test_loss:.4f}")
        
        return {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "loss": float(test_loss)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset_path> [output_path] [epochs] [batch_size]")
        print("\nExample:")
        print("  python train.py ./ml/data/crypto_trades.csv ./ml/models 50 32")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./ml/models"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    
    metrics = train_model(dataset_path, output_path, epochs, batch_size)
    
    print(f"\n📊 Final Metrics:")
    print(json.dumps(metrics, indent=2))
