# ML-Powered Solana Trading Bot - Complete Integration Guide

This guide covers the complete setup, deployment, and operation of the ML-enhanced trading bot.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Solana Network (gRPC)                     │
│                   (New Token Detection)                       │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Bot Core Logic │
                    │  (Node.js)      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │   Risk  │         │   ML    │        │Notif    │
    │Manager  │         │  Model  │        │Service  │
    │         │         │(Python) │        │         │
    └────┬────┘         └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │ Position│         │ Suggestion        │Dashboard│
    │ Manager │         │  Service          │API      │
    │         │         │  (Confidence)     │         │
    └─────────┘         └────┬────┘        └────┬────┘
                             │                   │
                    ┌────────▼───────────┐       │
                    │   User Interface   │◄──────┘
                    │  (Accept/Reject)   │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  Trade Executor    │
                    │  (Buy/Sell)        │
                    └────────────────────┘
```

## Complete Setup Steps

### Step 1: Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r ml/requirements.txt
```

**Python packages:**
- `tensorflow` - Deep learning framework
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `numpy` - Numerical computing
- `joblib` - Model serialization

### Step 2: Configure Environment Variables

Create `.env` file:

```bash
# Solana gRPC
GRPC_ENDPOINT=<your_grpc_endpoint>
GRPCTOKEN=<your_grpc_token>

# Wallet
PRIVATE_KEY=<your_wallet_private_key>
RPC_URL=<solana_rpc_url>

# Dashboard
ENABLE_DASHBOARD=true
DASHBOARD_PORT=3000

# Notifications
NOTIFICATION_CHANNEL=discord|telegram|email
```

### Step 3: Prepare and Download Dataset

```bash
# Create data directory
mkdir -p ml/data

# Download Kaggle dataset
# Option A: Via Kaggle CLI
kaggle datasets download -d <dataset_name> -p ml/data/

# Option B: Manual download
# Go to https://www.kaggle.com/ and download CSV manually
# Place in ml/data/your_dataset.csv
```

**Dataset Requirements:**

Minimum columns:
```csv
timestamp,close,volume,market_cap,profit
```

Expected format (for best results):
```csv
timestamp,open,high,low,close,volume,market_cap,
holders,liquidity,tx_count,volatility,profit
```

### Step 4: Train the ML Model

```bash
# Basic training (recommended: 50-100 epochs)
python ml/train.py ml/data/your_dataset.csv

# Custom configuration
python ml/train.py ml/data/your_dataset.csv ./ml/models 100 32

# With specific parameters
python ml/train.py \
  ml/data/crypto_trades.csv \
  ./ml/models \
  100 \
  32
```

**Training Output:**
```
============================================================
Solana Trade Prediction Model Training
============================================================

📊 Loading dataset from: ml/data/crypto_trades.csv
✓ Dataset loaded successfully
  Training samples: 8500
  Testing samples: 2125
  Features per sample: 10

📈 Dataset Statistics:
  total_samples: 10625
  features: 10
  missing_values: 0
  numeric_columns: 10
  shape: (10625, 10)

🤖 Building neural network model...

🚀 Training model for 100 epochs with batch size 32...
---------- training progress ----------
✓ Training completed!

📊 Evaluating on test set...
  Test Accuracy: 0.8245
  Test AUC: 0.8912
  Test Loss: 0.3456

💾 Saving model to: ./ml/models
✓ Model and metrics saved successfully!

============================================================
✅ Training Complete!
============================================================

Model ready for predictions.
Model location: ./ml/models
```

**Verify Training:**
```bash
# Check saved model
ls -la ml/models/
# Should contain:
# - trade_model.h5       (neural network)
# - scaler.pkl           (feature normalization)
# - metrics.json         (training metrics)

# View metrics
cat ml/models/metrics.json
```

### Step 5: Update Bot Configuration

Edit `config.js`:

```javascript
export const config = {
  trading: {
    sniperAmount: 1.0,
    profitTarget: 3,
    stopLoss: 0.5,
    maxHoldTime: 300000,
    maxPositions: 5,
    
    // Enable ML-based suggestions
    enableMLSuggestions: true,
    
    // Auto-accept high-confidence suggestions
    // Set to 0.85 to auto-accept 85%+ confidence
    // Set to null to require manual acceptance
    autoAcceptThreshold: 0.85,
  },
  
  risk: {
    maxDailyLoss: 10.0,
    maxDrawdown: 0.3,
    positionSizeLimit: 2.0,
  },
}
```

### Step 6: Start the Trading Bot

```bash
# Development mode (with hot reload)
npm run dev

# Production mode
npm start

# With logging
npm start 2>&1 | tee bot.log

# In background
nohup npm start &
```

**Expected Output:**
```
   ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗ 
  ██╔════╝██╔══██╗██║   ██║██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝ 
  ██║     ██████╔╝ ██║ ██╔╝██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ██╗ 
  ██║     ██╔══██╗   ██╔═╝ ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║
  ╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝
   ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 

🚀 Starting Solana Raydium Sniper Bot...
🔑 Wallet Public Key: 4xPpV...
💰 Sniper Amount: 1.0 SOL
🎯 Profit Target: 3x
🛑 Stop Loss: 0.5x
⏱️ Max Hold Time: 300s
📊 Max Positions: 5

Dashboard server started on port 3000
Dashboard available at: http://localhost:3000
```

### Step 7: Access Dashboard

Open in browser:
```
http://localhost:3000/dashboard/public/suggestions.html
```

**Dashboard Features:**
- View pending trade suggestions
- Accept/reject trades with one click
- Monitor execution history
- Track ML model performance
- View risk metrics
- Real-time statistics

## Operating the Bot

### Daily Operations

```bash
# 1. Check bot is running
curl http://localhost:3000/api/health

# 2. View pending suggestions
curl http://localhost:3000/api/suggestions/pending

# 3. View current positions
curl http://localhost:3000/api/positions

# 4. Monitor risk levels
curl http://localhost:3000/api/risk

# 5. Check daily stats
curl http://localhost:3000/api/stats
```

### Accepting/Rejecting Trades

**Via Dashboard UI:**
1. Open `http://localhost:3000/dashboard/public/suggestions.html`
2. View pending suggestions
3. Click "Accept Trade" or "Reject"
4. Monitor execution in real-time

**Via API:**

```bash
# Get pending suggestions
curl http://localhost:3000/api/suggestions/pending

# Accept a suggestion
curl -X POST http://localhost:3000/api/suggestions/{suggestionId}/accept

# Reject a suggestion
curl -X POST http://localhost:3000/api/suggestions/{suggestionId}/reject \
  -H "Content-Type: application/json" \
  -d '{"reason":"Risk too high"}'
```

### Executing Accepted Trades

Trades execute automatically when accepted (if properly configured):

```bash
# Check executing trades
curl http://localhost:3000/api/trades/executing

# View execution history
curl http://localhost:3000/api/trades/history?limit=20

# Get execution statistics
curl http://localhost:3000/api/trades/stats
```

## Retraining the Model

### Monthly Retraining

Create `cron` job:

```bash
# Edit crontab
crontab -e

# Add monthly retraining (1st of month at 2 AM)
0 2 1 * * cd /path/to/bot && python ml/train.py ml/data/new_data.csv
```

### Manual Retraining

```bash
# With new data
python ml/train.py ml/data/recent_trades.csv ./ml/models 100 32

# Verify new model
cat ml/models/metrics.json

# Restart bot to use new model
pkill -f "npm start"
npm start
```

### Backtest Before Deployment

```bash
# Create backtest dataset
# Split recent data 80/20 for training/testing

# Train on historical data
python ml/train.py ml/data/historical.csv ./ml/models

# Verify metrics are > 0.75 accuracy
cat ml/models/metrics.json

# Manual test on small trade amounts
# Set sniperAmount: 0.1 SOL for testing
```

## Monitoring & Debugging

### Check Bot Status

```bash
# View logs
tail -f bot.log

# Watch specific errors
grep "ERROR" bot.log | tail -20

# Monitor ML predictions
grep "Trade suggestion" bot.log
```

### Check Model Health

```bash
# Test model prediction
python -c "
from ml.model import TradePredictionModel
import numpy as np

model = TradePredictionModel(model_path='./ml/models')
test_data = np.random.randn(10)
result = model.predict(test_data)
print(result)
"
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `python ml/train.py ml/data/dataset.csv` |
| Low accuracy | Use larger dataset (1000+ samples) |
| Slow predictions | Reduce batch size or model complexity |
| Python errors | Check `pip install -r ml/requirements.txt` |
| gRPC connection | Verify GRPC_ENDPOINT and GRPCTOKEN |

## Performance Optimization

### Faster Training

```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0
python ml/train.py ml/data/dataset.csv ./ml/models 50 64

# Or with reduced epochs
python ml/train.py ml/data/dataset.csv ./ml/models 25 32
```

### Faster Predictions

```bash
# Reduce model complexity (edit ml/model.py)
# LSTM units: 128→64, 64→32, 32→16
# Saves 10-15ms per prediction
```

### Memory Management

```bash
# Monitor memory usage
watch -n 1 'free -h'

# If memory issues, reduce batch size
python ml/train.py dataset.csv ./ml/models 100 8
```

## Deployment Checklist

- [ ] Environment variables configured (.env)
- [ ] Dataset downloaded and verified
- [ ] Model trained with accuracy > 75%
- [ ] Small trade test successful (0.1 SOL)
- [ ] Dashboard accessible
- [ ] Risk management configured
- [ ] Notifications enabled
- [ ] Backup of trained model created
- [ ] Monitoring setup (logs, alerting)
- [ ] Retraining schedule established

## Advanced Configuration

### Risk Management

```javascript
// Adjust max daily loss
config.risk.maxDailyLoss = 5.0; // Stop trading if -5 SOL loss

// Set drawdown limit
config.risk.maxDrawdown = 0.25; // Stop at -25% portfolio drop

// Position size limit
config.risk.positionSizeLimit = 1.5; // Max 1.5x config amount
```

### Auto-Accept Thresholds

```javascript
// Aggressive: Auto-accept 80%+ confidence
config.trading.autoAcceptThreshold = 0.80;

// Conservative: Require manual acceptance
config.trading.autoAcceptThreshold = null;

// Balanced: Auto-accept 85%+ confidence
config.trading.autoAcceptThreshold = 0.85;
```

### Custom Notifications

```javascript
// Via Telegram
NOTIFICATION_CHANNEL=telegram
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHAT_ID=<chat_id>

// Via Discord
NOTIFICATION_CHANNEL=discord
DISCORD_WEBHOOK_URL=<webhook_url>

// Via Email
NOTIFICATION_CHANNEL=email
EMAIL_FROM=<your_email>
EMAIL_PASSWORD=<app_password>
EMAIL_TO=<recipient>
```

## Support & Resources

- **ML Setup**: See `ML_SETUP.md`
- **API Reference**: See dashboard endpoints documentation
- **Training Guide**: See `ml/README.md`
- **Bot Logs**: Check `bot.log` or `logs/` directory
- **Troubleshooting**: Review issues in logs and environment setup

## Next Steps

1. Complete initial setup and training
2. Run small test trades (0.1 SOL)
3. Monitor for 24-48 hours
4. Adjust thresholds based on results
5. Gradually increase trade amounts
6. Set up automated retraining
7. Establish monitoring and alerting
8. Document your configuration

## Summary

This integrated ML system provides:
- **Automated Predictions**: Deep learning model for trade profitability
- **User Control**: Manual accept/reject for all suggestions
- **Risk Management**: Built-in safety checks and position limits
- **Performance Tracking**: Detailed statistics and execution history
- **Easy Retraining**: Simple scripts for model updates
- **Production Ready**: Dashboard, APIs, and monitoring

Happy trading with ML-powered predictions!
