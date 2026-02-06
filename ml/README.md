# ML Trade Prediction System

This directory contains the deep learning model and dataset handling for intelligent trade suggestions on Solana tokens.

## Overview

The ML system uses an LSTM-based neural network to:
- Predict which token launches are likely to be profitable
- Provide confidence scores for each prediction
- Generate trade suggestions with ML-based reasoning
- Allow users to accept or reject suggestions before execution

## Architecture

```
ml/
├── model.py           # LSTM neural network for trade prediction
├── dataset_loader.py  # Kaggle dataset loading and preprocessing
├── train.py          # Training script for new models
├── requirements.txt  # Python dependencies
└── models/           # Trained model storage (auto-created)
```

## Setup

### 1. Install Python Dependencies

```bash
pip install -r ml/requirements.txt
```

Or with conda:
```bash
conda create -n solana-ml python=3.10
conda activate solana-ml
pip install -r ml/requirements.txt
```

### 2. Prepare Your Dataset

Download a Kaggle dataset with trading/crypto data:
- Option 1: Crypto trading data with historical prices
- Option 2: Token launch data from blockchain
- Option 3: DeFi protocol transaction data

Place your CSV file in the `ml/data/` directory.

**Required columns** (at minimum):
- `close` or `price` - Token price
- `volume` - Trading volume
- `timestamp` or `date` - Time information
- `profit` or `label` - Whether the trade was profitable (binary: 0 or 1)

### 3. Train the Model

```bash
# Basic training
python ml/train.py ml/data/your_dataset.csv

# With custom parameters
python ml/train.py ml/data/your_dataset.csv ./ml/models 100 32
```

Parameters:
- `dataset_path`: Path to your Kaggle CSV
- `output_path` (optional): Where to save the model (default: `./ml/models`)
- `epochs` (optional): Number of training epochs (default: 50)
- `batch_size` (optional): Batch size for training (default: 32)

### 4. Integration with Trading Bot

The trading bot automatically uses the ML model via the MLBridgeService:

```javascript
import mlBridge from './services/mlBridge.js';
import tradeSuggestionService from './services/tradeSuggestion.js';

// Generate a trade suggestion
const suggestion = await tradeSuggestionService.generateSuggestion(tokenData, config);

// User accepts or rejects
await tradeSuggestionService.acceptSuggestion(suggestion.id);
await tradeSuggestionService.rejectSuggestion(suggestion.id, 'Risk too high');
```

## Model Details

### Network Architecture

```
Input Layer (10 features)
    ↓
LSTM (128 units) → Dropout (0.2)
    ↓
LSTM (64 units) → Dropout (0.2)
    ↓
LSTM (32 units)
    ↓
Dense (64 units) → Dropout (0.2)
    ↓
Dense (32 units)
    ↓
Output Layer (sigmoid) → Profitability prediction
```

### Input Features

The model expects these 10 features for each token:
1. **volume** - 24h trading volume (log-normalized)
2. **liquidity** - Available liquidity in SOL
3. **holder_count** - Number of token holders
4. **tx_count** - Number of transactions
5. **price_change_1m** - Price change in last 1 minute (%)
6. **price_change_5m** - Price change in last 5 minutes (%)
7. **volatility** - Price volatility (standard deviation)
8. **market_cap** - Token market cap (log-normalized)
9. **created_timestamp** - Token creation time
10. **dev_activity** - Development activity indicator

### Output

The model outputs:
- **prediction** (0-1): Probability that the trade is profitable
- **confidence** (0-1): Model confidence in the prediction
- **profitable** (boolean): `true` if probability > 0.5

## Trading Workflow

### 1. New Token Detected
```
gRPC Stream → Token Data
```

### 2. ML Prediction
```
Token Data → ML Model → Confidence Score
```

### 3. Suggestion Generation
```
Confidence Score + Rules → Trade Suggestion
    ├─ Token Info
    ├─ Action (BUY/HOLD)
    ├─ Amount to Trade
    └─ Reasoning
```

### 4. User Decision
```
User Accepts ──→ Execute Trade
User Rejects ──→ Log & Continue Monitoring
Expires ───────→ Auto-reject
```

## API Reference

### MLBridgeService

```javascript
// Predict if a trade will be profitable
await mlBridge.predictTrade(tokenData);
// Returns: { profitable, confidence, probability, status }

// Train model with new dataset
await mlBridge.trainModel(datasetPath);

// Get model statistics
await mlBridge.getModelStats();
```

### TradeSuggestionService

```javascript
// Generate a trade suggestion
await tradeSuggestionService.generateSuggestion(tokenData, config);
// Returns: { success, suggestion }

// Accept a suggestion
await tradeSuggestionService.acceptSuggestion(suggestionId);

// Reject a suggestion
await tradeSuggestionService.rejectSuggestion(suggestionId, 'reason');

// Get all pending suggestions
tradeSuggestionService.getPendingSuggestions();

// Get suggestion history
tradeSuggestionService.getSuggestionHistory(20);

// Get statistics
tradeSuggestionService.getStatistics();
```

## Performance Metrics

After training, check `ml/models/metrics.json` for:
- Test Accuracy
- Test AUC (Area Under Curve)
- Test Loss
- Training configuration

## Troubleshooting

### Model Not Found
If you see "ML model not found" messages:
1. Run the training script: `python ml/train.py`
2. Ensure models are saved in `ml/models/`

### Python Import Errors
```bash
# Make sure dependencies are installed
pip install -r ml/requirements.txt

# Or use conda environment
conda activate solana-ml
```

### Memory Issues During Training
Reduce batch size:
```bash
python ml/train.py dataset.csv ./ml/models 50 16
```

## Improving Predictions

1. **Use More Data**: Larger datasets → Better predictions
2. **More Epochs**: Increase training epochs (50-200)
3. **Feature Engineering**: Add custom indicators
4. **Cross-Validation**: Validate on multiple datasets
5. **Regular Retraining**: Retrain monthly with new data

## Example Workflow

```bash
# 1. Download Kaggle crypto trading dataset
# (Store in ml/data/trades.csv)

# 2. Train model
python ml/train.py ml/data/trades.csv ./ml/models 100 32

# 3. Start trading bot
npm start

# 4. Bot will now generate ML-based trade suggestions
# User can accept/reject each suggestion
```

## Future Improvements

- [ ] Ensemble models (multiple networks)
- [ ] Real-time model retraining
- [ ] Feature importance analysis
- [ ] Backtesting framework
- [ ] Web interface for suggestions
- [ ] Risk-adjusted position sizing
- [ ] Multi-token portfolio optimization

## Support

For issues with:
- **Model Training**: Check `ml/data/` for dataset format
- **Predictions**: Verify feature values in token data
- **Integration**: Check `services/mlBridge.js` logs
