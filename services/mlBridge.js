import { spawn } from 'child_process';
import path from 'path';
import logger from '../utils/logger.js';
import fs from 'fs';

/**
 * ML Bridge Service
 * Communicates with Python ML model for trade predictions
 */
class MLBridgeService {
  constructor() {
    this.pythonModelPath = path.join(process.cwd(), 'ml', 'model.py');
    this.modelLoaded = false;
    this.predictionCache = new Map();
    this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
  }

  /**
   * Get trade prediction for a token
   * @param {Object} tokenData - Market data for the token
   * @returns {Promise<Object>} Prediction result with confidence
   */
  async predictTrade(tokenData) {
    try {
      // Check cache first
      const cacheKey = this._getCacheKey(tokenData);
      const cachedResult = this._getFromCache(cacheKey);
      if (cachedResult) {
        logger.debug('ML prediction from cache', { token: tokenData.symbol });
        return cachedResult;
      }

      // Extract features for model
      const features = {
        volume: tokenData.volume || 0,
        liquidity: tokenData.liquidity || 0,
        holder_count: tokenData.holders || 0,
        tx_count: tokenData.txCount || 0,
        price_change_1m: tokenData.priceChange1m || 0,
        price_change_5m: tokenData.priceChange5m || 0,
        volatility: tokenData.volatility || 0.1,
        market_cap: tokenData.marketCap || 0,
        created_timestamp: tokenData.createdAt || Date.now(),
        dev_activity: tokenData.devActivity || 0
      };

      // Call Python model
      const prediction = await this._callPythonModel(features);

      if (!prediction.error) {
        // Cache the result
        this._setCache(cacheKey, prediction);
        
        logger.info('ML prediction generated', {
          token: tokenData.symbol,
          profitable: prediction.profitable,
          confidence: prediction.confidence
        });
      }

      return prediction;
    } catch (error) {
      logger.error('Error in trade prediction', error);
      return {
        error: error.message,
        profitable: false,
        confidence: 0,
        status: 'failed'
      };
    }
  }

  /**
   * Call Python model via subprocess
   * @private
   */
  _callPythonModel(features) {
    return new Promise((resolve, reject) => {
      try {
        // Check if Python script exists
        if (!fs.existsSync(this.pythonModelPath)) {
          logger.warn('ML model not found, returning neutral prediction');
          return resolve({
            profitable: false,
            confidence: 0.5,
            probability: 0.5,
            status: 'model_not_found'
          });
        }

        console.log("Spawning Python model with features:", Object.keys(features));
        
        const python = spawn('python3', [
          this.pythonModelPath,
          'predict',
          JSON.stringify(features)
        ]);

        let output = '';
        let errorOutput = '';

        python.stdout.on('data', (data) => {
          console.log("Python stdout:", data.toString());
          output += data.toString();
        });

        python.stderr.on('data', (data) => {
          console.log("Python stderr:", data.toString());
          errorOutput += data.toString();
        });

        python.on('close', (code) => {
          console.log("Python process closed with code:", code);
          if (code !== 0) {
            logger.warn('Python model error:', errorOutput);
            console.log("ML prediction failed, error:", errorOutput);
            return resolve({
              error: errorOutput,
              profitable: false,
              confidence: 0,
              status: 'failed'
            });
          }

          try {
            console.log("Parsing ML output:", output);
            const result = JSON.parse(output);
            console.log("ML prediction result:", result);
            resolve(result);
          } catch (parseError) {
            logger.error('Failed to parse model output', parseError);
            resolve({
              error: 'Failed to parse model output',
              profitable: false,
              confidence: 0,
              status: 'failed'
            });
          }
        });

        // Timeout after 10 seconds
        setTimeout(() => {
          python.kill();
          resolve({
            error: 'Model prediction timeout',
            profitable: false,
            confidence: 0,
            status: 'timeout'
          });
        }, 10000);
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Generate cache key from token data
   * @private
   */
  _getCacheKey(tokenData) {
    return `${tokenData.symbol || tokenData.address}:${Math.floor(Date.now() / 60000)}`;
  }

  /**
   * Get cached prediction if still valid
   * @private
   */
  _getFromCache(key) {
    const cached = this.predictionCache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
      return cached.result;
    }
    this.predictionCache.delete(key);
    return null;
  }

  /**
   * Set cache entry
   * @private
   */
  _setCache(key, result) {
    this.predictionCache.set(key, {
      result,
      timestamp: Date.now()
    });
  }

  /**
   * Clear old cache entries
   */
  clearExpiredCache() {
    const now = Date.now();
    for (const [key, value] of this.predictionCache.entries()) {
      if (now - value.timestamp > this.cacheExpiry) {
        this.predictionCache.delete(key);
      }
    }
  }

  /**
   * Train model with new dataset
   * @param {string} datasetPath - Path to Kaggle CSV dataset
   * @returns {Promise<Object>} Training results
   */
  async trainModel(datasetPath) {
    return new Promise((resolve, reject) => {
      try {
        const datasetLoaderPath = path.join(process.cwd(), 'ml', 'dataset_loader.py');
        
        if (!fs.existsSync(datasetLoaderPath)) {
          return resolve({
            error: 'Dataset loader not found',
            status: 'failed'
          });
        }

        const python = spawn('python3', [datasetLoaderPath, datasetPath]);

        let output = '';
        let errorOutput = '';

        python.stdout.on('data', (data) => {
          output += data.toString();
          logger.info('Model training progress:', data.toString());
        });

        python.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });

        python.on('close', (code) => {
          if (code !== 0) {
            logger.error('Training failed:', errorOutput);
            return resolve({
              error: errorOutput,
              status: 'failed'
            });
          }

          resolve({
            status: 'success',
            message: 'Model trained successfully',
            output
          });
        });

        // Training timeout (30 minutes)
        setTimeout(() => {
          python.kill();
          resolve({
            error: 'Training timeout',
            status: 'timeout'
          });
        }, 30 * 60 * 1000);
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Get model statistics
   */
  async getModelStats() {
    return {
      cacheSize: this.predictionCache.size,
      cacheExpiry: this.cacheExpiry,
      modelPath: this.pythonModelPath,
      modelExists: fs.existsSync(this.pythonModelPath)
    };
  }
}

// Export singleton instance
const mlBridge = new MLBridgeService();
export default mlBridge;
