# ML-Powered Solana Trading Bot - Setup Guide

This guide walks you through integrating deep learning predictions into your Solana trading bot.

## Quick Start

### 1. Install Python Dependencies

```bash
# Install Python packages for ML
pip install -r ml/requirements.txt
```

### 2. Prepare Your Dataset

The ML model needs historical trading data from Kaggle or similar sources.

**Where to find datasets:**
- [Kaggle Crypto Datasets](https://www.kaggle.com/search?q=crypto+trading)
- [Kaggle DeFi Data](https://www.kaggle.com/search?q=defi)
- [Binance Historical Data](https://www.kaggle.com/code/stefano99/trading-data)

**Example datasets:**
- "Cryptocurrency Historical Data" - OHLCV data for multiple cryptos
- "Solana Token Data" - Solana-specific trading metrics
- "DeFi Protocol Transactions" - Decentralized finance activity data

### 3. Download and Place Dataset

```bash
# Create data directory if it doesn't exist
mkdir -p ml/data

# Download your chosen dataset from Kaggle
# Place the CSV file in ml/data/
# Example: ml/data/crypto_trades.csv
```

**Dataset Requirements:**

Your CSV should have these columns (or similar):
```
timestamp,open,high,low,close,volume,market_cap,holders
```

Or for pre-trained model to work directly:
```
volume,liquidity,holder_count,tx_count,price_change_1m,
price_change_5m,volatility,market_cap,created_timestamp,
dev_activity,profit
```

### 4. Train the Model

```bash
# Basic training (50 epochs)
python ml/train.py ml/data/your_dataset.csv

# Custom training (100 epochs, batch size 16)
python ml/train.py ml/data/your_dataset.csv ./ml/models 100 16
```

**What to expect:**
- Training takes 5-30 minutes depending on dataset size
- Model is saved to `ml/models/`
- Training metrics saved to `ml/models/metrics.json`

### 5. Run the Trading Bot

```bash
# With ML predictions enabled
npm start

# Dashboard will show trade suggestions
# Accept or reject each suggested trade
```

## How It Works

### Data Flow

```
New Token Detected
        ↓
Extract Market Data
        ↓
ML Model Prediction
        ↓
Generate Suggestion (with confidence)
        ↓
User Accepts/Rejects
        ↓
Execute Trade (if accepted)
```

### Model Architecture

```
Token Market Data (10 features)
        ↓
LSTM Network (128→64→32 units)
        ↓
Dense Layers (64→32 units)
        ↓
Output: Profitability Probability (0-1)
```

## Configuration

### Enable ML Features in `config.js`

```javascript
export const config = {
  trading: {
    // ... existing config ...
    
    // Enable ML-based suggestions
    enableMLSuggestions: true,
    
    // Auto-accept if confidence > threshold (0.0-1.0)
    // Set to null to require manual acceptance
    autoAcceptThreshold: 0.85,
  }
}
```

### Dashboard API Endpoints

Once running, the dashboard provides these ML endpoints:

```
GET  /api/suggestions/pending          - Get pending trade suggestions
GET  /api/suggestions/:id              - Get specific suggestion
POST /api/suggestions/:id/accept       - Accept a trade suggestion
POST /api/suggestions/:id/reject       - Reject with reason
GET  /api/suggestions/stats            - Get acceptance statistics
GET  /api/suggestions/history?limit=20 - Get historical suggestions
GET  /api/ml/stats                     - Get ML model statistics
```

### Example: Manual Trade Control

```bash
# Check pending suggestions
curl http://localhost:3000/api/suggestions/pending

# Accept a suggestion
curl -X POST http://localhost:3000/api/suggestions/sug_123456/accept

# Reject a suggestion
curl -X POST http://localhost:3000/api/suggestions/sug_123456/reject \
  -H "Content-Type: application/json" \
  -d '{"reason":"Risk too high"}'
```

## Feature Engineering

The model uses 10 features automatically extracted from each token:

1. **volume** - 24h trading volume (log-normalized)
2. **liquidity** - Available pool liquidity
3. **holder_count** - Number of token holders
4. **tx_count** - Number of transactions
5. **price_change_1m** - 1-minute price change %
6. **price_change_5m** - 5-minute price change %
7. **volatility** - Price volatility metric
8. **market_cap** - Token market cap (log-normalized)
9. **created_timestamp** - Token creation timestamp
10. **dev_activity** - Development activity level

## Improving Model Performance

### 1. Better Dataset

```bash
# Use larger, more recent datasets (1000+ samples)
# Verify data quality (no missing values)
# Ensure balanced classes (profit/loss distribution)
```

### 2. Hyperparameter Tuning

```bash
# Increase epochs for better convergence
python ml/train.py data.csv ./ml/models 200 32

# Reduce batch size for noisy gradients
python ml/train.py data.csv ./ml/models 100 16

# Experiment with different values
python ml/train.py data.csv ./ml/models 150 64
```

### 3. Regular Retraining

Schedule monthly retraining with new data:

```bash
# Add to cron (monthly retraining)
0 0 1 * * cd /path/to/bot && python ml/train.py ml/data/new_data.csv
```

## Troubleshooting

### Python Not Found

```bash
# Ensure Python 3.8+ is installed
python3 --version

# Or specify full path
/usr/bin/python3 ml/train.py dataset.csv
```

### Memory Issues

```bash
# Reduce batch size
python ml/train.py dataset.csv ./ml/models 50 8

# Or reduce model complexity (edit ml/model.py)
```

### Low Accuracy

```bash
# Check dataset quality
python ml/dataset_loader.py ml/data/dataset.csv

# Verify target column exists (named 'profit')
# Ensure balanced class distribution
```

### Model Not Loading

```bash
# Verify files exist
ls -la ml/models/

# Should have:
# - trade_model.h5
# - scaler.pkl
# - metrics.json

# If missing, retrain:
python ml/train.py ml/data/dataset.csv
```

## Advanced Usage

### Custom Feature Engineering

Edit `ml/dataset_loader.py` `_engineer_features()` method:

```python
def _engineer_features(self):
    # Add your own indicators
    self.df['rsi'] = calculate_rsi(self.df['close'])
    self.df['macd'] = calculate_macd(self.df['close'])
    # ...
```

### Ensemble Models

Combine multiple models for better predictions:

```python
from ml.model import TradePredictionModel

model1 = TradePredictionModel(model_path="./ml/models/model1")
model2 = TradePredictionModel(model_path="./ml/models/model2")

# Average predictions
avg_prediction = (model1.predict(data) + model2.predict(data)) / 2
```

### Model Evaluation

```bash
# After training, check metrics
cat ml/models/metrics.json

# Expected output:
{
  "test_accuracy": 0.82,
  "test_auc": 0.89,
  "test_loss": 0.34,
  "epochs": 50
}
```

## Production Checklist

- [ ] Dataset prepared and validated
- [ ] Model trained with > 80% accuracy
- [ ] Dashboard endpoints tested
- [ ] Trade suggestions manually verified
- [ ] Risk management properly configured
- [ ] Notifications enabled
- [ ] Backup of trained model created
- [ ] Retraining schedule established

## Environment Variables

No additional environment variables needed. ML runs with existing bot config.

Optional: Add to `.env` for custom model paths:
```
ML_MODEL_PATH=./ml/models
ML_ENABLE_SUGGESTIONS=true
```

## Next Steps

1. ✅ Complete ML setup
2. ✅ Train your first model
3. ✅ Run bot and accept/reject suggestions
4. ✅ Review performance metrics
5. ✅ Adjust thresholds based on results
6. ✅ Set up automated retraining

## Support & Resources

- **ML Documentation**: See `ml/README.md`
- **Training Script Help**: `python ml/train.py --help`
- **Dataset Issues**: Verify CSV format matches expectations
- **Model Errors**: Check Python version and dependencies

## Example Workflow

```bash
# 1. Download Kaggle dataset
wget https://www.kaggle.com/.../crypto_data.csv -O ml/data/trades.csv

# 2. Verify dataset
python ml/dataset_loader.py ml/data/trades.csv

# 3. Train model (100 epochs)
python ml/train.py ml/data/trades.csv ./ml/models 100 32

# 4. Start bot
npm start

# 5. Monitor suggestions via dashboard
open http://localhost:3000

# 6. Accept/reject trades manually
curl http://localhost:3000/api/suggestions/pending

# 7. Check performance after 24 hours
curl http://localhost:3000/api/suggestions/stats
```

Happy trading with ML!
