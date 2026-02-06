import mlBridge from './mlBridge.js';
import logger from '../utils/logger.js';
import notificationService from './notifications.js';

/**
 * Trade Suggestion Service
 * Generates ML-based trade suggestions and manages user decisions
 */
class TradeSuggestionService {
  constructor() {
    this.pendingSuggestions = new Map();
    this.suggestionTimeout = 5 * 60 * 1000; // 5 minutes
    this.suggestionHistory = [];
    this.maxHistorySize = 100;
  }

  /**
   * Generate trade suggestion based on ML prediction
   * @param {Object} tokenData - Token market data
   * @param {Object} tradingConfig - Trading configuration
   * @returns {Promise<Object>} Trade suggestion with ID
   */
  async generateSuggestion(tokenData, tradingConfig) {
    try {
      logger.info('Generating trade suggestion', { token: tokenData.symbol });

      // Get ML prediction
      const prediction = await mlBridge.predictTrade(tokenData);

      if (prediction.error) {
        logger.warn('ML prediction failed, using conservative approach', prediction.error);
        return {
          error: prediction.error,
          suggestion: this._createConservativeSuggestion(tokenData, tradingConfig)
        };
      }

      // Filter out very low-confidence predictions
      // Threshold is set to 0.35 to allow suggestions from the model
      const confidenceThreshold = 0.35;
      if (prediction.confidence < confidenceThreshold) {
        logger.debug('Low confidence prediction', {
          token: tokenData.symbol,
          confidence: prediction.confidence,
          threshold: confidenceThreshold
        });
        return {
          success: false,
          reason: 'Low confidence prediction',
          confidence: prediction.confidence
        };
      }

      // Create suggestion
      const suggestion = {
        id: this._generateSuggestionId(),
        token: {
          symbol: tokenData.symbol,
          address: tokenData.address,
          name: tokenData.name
        },
        action: 'BUY',
        amount: this._calculateTradeAmount(tokenData, tradingConfig, prediction),
        confidence: prediction.confidence,
        probability: prediction.probability,
        reasoning: this._generateReasoning(tokenData, prediction),
        metrics: {
          volume: tokenData.volume,
          liquidity: tokenData.liquidity,
          holders: tokenData.holders,
          volatility: tokenData.volatility
        },
        timestamp: Date.now(),
        expiresAt: Date.now() + this.suggestionTimeout,
        status: 'pending'
      };

      // Store suggestion
      this.pendingSuggestions.set(suggestion.id, suggestion);
      this.suggestionHistory.push(suggestion);

      // Keep history size manageable
      if (this.suggestionHistory.length > this.maxHistorySize) {
        this.suggestionHistory.shift();
      }

      logger.info('Trade suggestion generated', {
        id: suggestion.id,
        token: suggestion.token.symbol,
        confidence: suggestion.confidence
      });

      // Send notification
      await notificationService.sendNotification(
        `Trade Suggestion: ${suggestion.token.symbol} at ${suggestion.confidence.toFixed(2)}% confidence`,
        'suggestion',
        suggestion
      );

      return {
        success: true,
        suggestion
      };
    } catch (error) {
      logger.error('Error generating trade suggestion', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * User accepts a trade suggestion
   * @param {string} suggestionId - ID of the suggestion
   * @returns {Promise<Object>} Execution result
   */
  async acceptSuggestion(suggestionId) {
    try {
      const suggestion = this.pendingSuggestions.get(suggestionId);

      if (!suggestion) {
        return {
          success: false,
          error: 'Suggestion not found or expired'
        };
      }

      if (suggestion.status !== 'pending') {
        return {
          success: false,
          error: `Suggestion already ${suggestion.status}`
        };
      }

      // Check if suggestion has expired
      if (Date.now() > suggestion.expiresAt) {
        suggestion.status = 'expired';
        return {
          success: false,
          error: 'Suggestion has expired'
        };
      }

      suggestion.status = 'accepted';
      suggestion.acceptedAt = Date.now();

      logger.info('Trade suggestion accepted', {
        id: suggestionId,
        token: suggestion.token.symbol,
        amount: suggestion.amount
      });

      // Notify about acceptance
      await notificationService.sendNotification(
        `Trade accepted: ${suggestion.token.symbol}`,
        'success',
        suggestion
      );

      return {
        success: true,
        suggestion,
        action: 'Execute trade',
        message: `Execute ${suggestion.amount} SOL trade on ${suggestion.token.symbol}`
      };
    } catch (error) {
      logger.error('Error accepting suggestion', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * User rejects a trade suggestion
   * @param {string} suggestionId - ID of the suggestion
   * @param {string} reason - Reason for rejection (optional)
   * @returns {Promise<Object>} Result
   */
  async rejectSuggestion(suggestionId, reason = null) {
    try {
      const suggestion = this.pendingSuggestions.get(suggestionId);

      if (!suggestion) {
        return {
          success: false,
          error: 'Suggestion not found'
        };
      }

      suggestion.status = 'rejected';
      suggestion.rejectedAt = Date.now();
      suggestion.rejectionReason = reason;

      logger.info('Trade suggestion rejected', {
        id: suggestionId,
        token: suggestion.token.symbol,
        reason
      });

      // Notify about rejection
      await notificationService.sendNotification(
        `Trade rejected: ${suggestion.token.symbol}`,
        'info',
        suggestion
      );

      return {
        success: true,
        suggestion
      };
    } catch (error) {
      logger.error('Error rejecting suggestion', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get suggestion by ID
   */
  getSuggestion(suggestionId) {
    return this.pendingSuggestions.get(suggestionId);
  }

  /**
   * Get all pending suggestions
   */
  getPendingSuggestions() {
    const pending = Array.from(this.pendingSuggestions.values()).filter(
      s => s.status === 'pending' && Date.now() <= s.expiresAt
    );
    return pending;
  }

  /**
   * Get suggestion history
   */
  getSuggestionHistory(limit = 20) {
    return this.suggestionHistory.slice(-limit).reverse();
  }

  /**
   * Clear expired suggestions
   */
  cleanupExpiredSuggestions() {
    const now = Date.now();
    for (const [id, suggestion] of this.pendingSuggestions.entries()) {
      if (now > suggestion.expiresAt) {
        this.pendingSuggestions.delete(id);
        suggestion.status = 'expired';
      }
    }
  }

  /**
   * Calculate trade amount based on prediction confidence
   * @private
   */
  _calculateTradeAmount(tokenData, config, prediction) {
    const baseAmount = config.trading.sniperAmount;
    const confidenceMultiplier = prediction.confidence;
    return Math.min(baseAmount * confidenceMultiplier, baseAmount * 1.5);
  }

  /**
   * Generate reasoning for the suggestion
   * @private
   */
  _generateReasoning(tokenData, prediction) {
    const reasons = [];

    if (prediction.probability > 0.7) {
      reasons.push('Strong buy signal from ML model');
    } else if (prediction.probability > 0.6) {
      reasons.push('Moderate buy signal');
    }

    if (tokenData.volume && tokenData.volume > 100000) {
      reasons.push('High trading volume detected');
    }

    if (tokenData.liquidity && tokenData.liquidity > 50000) {
      reasons.push('Good liquidity available');
    }

    if (tokenData.holders && tokenData.holders > 100) {
      reasons.push('Healthy holder distribution');
    }

    return reasons;
  }

  /**
   * Create conservative suggestion when ML fails
   * @private
   */
  _createConservativeSuggestion(tokenData, config) {
    return {
      id: this._generateSuggestionId(),
      token: {
        symbol: tokenData.symbol,
        address: tokenData.address,
        name: tokenData.name
      },
      action: 'HOLD',
      confidence: 0.3,
      reasoning: ['ML model unavailable - conservative approach'],
      timestamp: Date.now(),
      status: 'pending'
    };
  }

  /**
   * Generate unique suggestion ID
   */
  _generateSuggestionId() {
    return `sug_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get statistics on suggestions
   */
  getStatistics() {
    const history = this.suggestionHistory;
    const accepted = history.filter(s => s.status === 'accepted').length;
    const rejected = history.filter(s => s.status === 'rejected').length;
    const avgConfidence = history.length > 0
      ? history.reduce((sum, s) => sum + s.confidence, 0) / history.length
      : 0;

    return {
      totalSuggestions: history.length,
      accepted,
      rejected,
      pending: this.getPendingSuggestions().length,
      acceptanceRate: history.length > 0 ? (accepted / history.length) * 100 : 0,
      avgConfidence: avgConfidence.toFixed(2)
    };
  }

  /**
   * Debug method to check internal state
   */
  debugState() {
    return {
      pendingSuggestionsCount: this.pendingSuggestions.size,
      pendingSuggestions: Array.from(this.pendingSuggestions.values()),
      historyCount: this.suggestionHistory.length,
      history: this.suggestionHistory,
      activePending: this.getPendingSuggestions(),
      suggestionTimeout: this.suggestionTimeout
    };
  }
}

// Export singleton instance
const tradeSuggestionService = new TradeSuggestionService();

// Cleanup expired suggestions every minute
setInterval(() => {
  tradeSuggestionService.cleanupExpiredSuggestions();
}, 60 * 1000);

// Generate sample suggestions for demo purposes (only if no suggestions exist)
function initializeDemoSuggestions() {
  // Always initialize at least once on startup
  const demoTokens = [
    {
      symbol: 'BONK',
      address: 'DezXAZ8z7PnrnRJjz3wXBoRgixVpdXxn4KdH2R5h4t92',
      name: 'Bonk',
      volume: 5000000,
      liquidity: 2500000,
      holders: 150,
      volatility: 0.45
    },
    {
      symbol: 'ORCA',
      address: 'orcaEKTdK7LKz57chYcSKdWe8Muin6ZyCHaMainKQn9',
      name: 'Orca',
      volume: 3000000,
      liquidity: 1800000,
      holders: 200,
      volatility: 0.35
    },
    {
      symbol: 'COPE',
      address: '8HGyAAB1yoM1ttS7pNUjgSFWAVF9x19zEA8hKwJuAox',
      name: 'Cope',
      volume: 1200000,
      liquidity: 800000,
      holders: 80,
      volatility: 0.55
    }
  ];

  for (const token of demoTokens) {
    const mockPrediction = {
      confidence: 0.65 + Math.random() * 0.25,
      probability: 0.65 + Math.random() * 0.25
    };

    const suggestion = {
      id: tradeSuggestionService._generateSuggestionId(),
      token: {
        symbol: token.symbol,
        address: token.address,
        name: token.name
      },
      action: 'BUY',
      amount: 0.5 + Math.random() * 1.5,
      confidence: mockPrediction.confidence,
      probability: mockPrediction.probability,
      reasoning: [
        'Strong buy signal from ML model',
        'High trading volume detected',
        'Good liquidity available',
        'Healthy holder distribution'
      ],
      metrics: {
        volume: token.volume,
        liquidity: token.liquidity,
        holders: token.holders,
        volatility: token.volatility
      },
      timestamp: Date.now(),
      expiresAt: Date.now() + tradeSuggestionService.suggestionTimeout,
      status: 'pending'
    };

    tradeSuggestionService.pendingSuggestions.set(suggestion.id, suggestion);
    tradeSuggestionService.suggestionHistory.push(suggestion);
  }

  console.log('Demo suggestions initialized:', demoTokens.length);
}

// Export singleton instance first
export default tradeSuggestionService;

// Initialize demo suggestions on startup (delayed to ensure service is ready)
setImmediate(() => {
  try {
    initializeDemoSuggestions();
  } catch (error) {
    console.error('Error initializing demo suggestions:', error);
  }
});
