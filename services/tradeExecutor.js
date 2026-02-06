import { token_buy, token_sell } from '../fuc.js';
import riskManager from './riskManager.js';
import logger from '../utils/logger.js';
import notificationService from './notifications.js';
import tradeSuggestionService from './tradeSuggestion.js';

/**
 * Trade Executor Service
 * Handles execution of ML-approved trades
 */
class TradeExecutor {
  constructor() {
    this.executingTrades = new Map();
    this.executedTrades = [];
    this.maxHistorySize = 100;
  }

  /**
   * Execute a trade based on accepted ML suggestion
   * @param {string} suggestionId - ML suggestion ID
   * @param {Object} tradingConfig - Trading configuration
   * @returns {Promise<Object>} Execution result
   */
  async executeFromSuggestion(suggestionId, tradingConfig) {
    try {
      const suggestion = tradeSuggestionService.getSuggestion(suggestionId);

      if (!suggestion) {
        return {
          success: false,
          error: 'Suggestion not found'
        };
      }

      if (suggestion.status !== 'accepted') {
        return {
          success: false,
          error: `Cannot execute: Suggestion status is ${suggestion.status}`
        };
      }

      // Check if already executing
      if (this.executingTrades.has(suggestionId)) {
        return {
          success: false,
          error: 'Trade already executing'
        };
      }

      // Mark as executing
      this.executingTrades.set(suggestionId, {
        startTime: Date.now(),
        status: 'executing'
      });

      logger.info('Executing trade from ML suggestion', {
        suggestionId,
        token: suggestion.token.symbol,
        amount: suggestion.amount,
        confidence: suggestion.confidence
      });

      try {
        // Check risk management rules
        const riskCheck = riskManager.canExecuteTrade(
          suggestion.amount,
          suggestion.token.address
        );

        if (!riskCheck.allowed) {
          logger.warn('Trade blocked by risk management', {
            suggestionId,
            reasons: riskCheck.errors
          });

          return {
            success: false,
            error: 'Trade blocked by risk management',
            reasons: riskCheck.errors
          };
        }

        // Execute the buy trade
        logger.info('Executing buy trade', {
          token: suggestion.token.address,
          amount: suggestion.amount
        });

        const buyResult = await token_buy(
          suggestion.token.address,
          suggestion.amount
        );

        if (!buyResult || !buyResult.txHash) {
          throw new Error('Buy trade failed');
        }

        const tradeRecord = {
          id: `trade_${Date.now()}`,
          suggestionId,
          token: suggestion.token,
          amount: suggestion.amount,
          confidence: suggestion.confidence,
          type: 'buy',
          txHash: buyResult.txHash,
          executedAt: Date.now(),
          status: 'open'
        };

        // Record the trade
        riskManager.recordTrade(
          'buy',
          suggestion.token.address,
          suggestion.amount,
          buyResult.price || 0,
          buyResult.txHash
        );

        // Track the position
        riskManager.activePositions.set(suggestion.token.address, {
          mint: suggestion.token.address,
          symbol: suggestion.token.symbol,
          entryAmount: suggestion.amount,
          entryPrice: buyResult.price || 0,
          entryTime: Date.now(),
          currentPrice: buyResult.price || 0,
          confidence: suggestion.confidence,
          txHash: buyResult.txHash
        });

        this.executedTrades.push(tradeRecord);
        if (this.executedTrades.length > this.maxHistorySize) {
          this.executedTrades.shift();
        }

        logger.info('Buy trade executed successfully', {
          token: suggestion.token.symbol,
          txHash: buyResult.txHash,
          amount: suggestion.amount
        });

        // Send notification
        await notificationService.sendNotification(
          `Buy trade executed: ${suggestion.token.symbol} for ${suggestion.amount} SOL`,
          'success',
          tradeRecord
        );

        this.executingTrades.delete(suggestionId);

        return {
          success: true,
          message: 'Trade executed successfully',
          trade: tradeRecord,
          txHash: buyResult.txHash
        };

      } catch (error) {
        logger.error('Trade execution failed', {
          suggestionId,
          error: error.message
        });

        this.executingTrades.delete(suggestionId);

        // Send error notification
        await notificationService.sendNotification(
          `Trade execution failed: ${suggestion.token.symbol} - ${error.message}`,
          'error',
          { suggestionId, error: error.message }
        );

        return {
          success: false,
          error: error.message
        };
      }

    } catch (error) {
      logger.error('Error in executeFromSuggestion', error);
      this.executingTrades.delete(suggestionId);

      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute a sell order for a position
   * @param {string} tokenAddress - Token mint address
   * @param {number} amount - Amount to sell
   * @returns {Promise<Object>} Sell result
   */
  async executeSell(tokenAddress, amount) {
    try {
      logger.info('Executing sell trade', {
        token: tokenAddress,
        amount
      });

      const sellResult = await token_sell(tokenAddress, amount);

      if (!sellResult || !sellResult.txHash) {
        throw new Error('Sell trade failed');
      }

      // Record the trade
      riskManager.recordTrade(
        'sell',
        tokenAddress,
        amount,
        sellResult.price || 0,
        sellResult.txHash
      );

      // Remove from active positions
      riskManager.activePositions.delete(tokenAddress);

      const tradeRecord = {
        id: `trade_${Date.now()}`,
        token: tokenAddress,
        type: 'sell',
        amount,
        txHash: sellResult.txHash,
        executedAt: Date.now(),
        price: sellResult.price || 0
      };

      this.executedTrades.push(tradeRecord);
      if (this.executedTrades.length > this.maxHistorySize) {
        this.executedTrades.shift();
      }

      logger.info('Sell trade executed', {
        token: tokenAddress,
        txHash: sellResult.txHash
      });

      return {
        success: true,
        trade: tradeRecord,
        txHash: sellResult.txHash
      };

    } catch (error) {
      logger.error('Sell trade execution failed', error);

      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get execution history
   */
  getExecutionHistory(limit = 20) {
    return this.executedTrades.slice(-limit).reverse();
  }

  /**
   * Get trades currently executing
   */
  getExecutingTrades() {
    return Array.from(this.executingTrades.entries()).map(([id, trade]) => ({
      suggestionId: id,
      ...trade
    }));
  }

  /**
   * Get execution statistics
   */
  getStatistics() {
    const buys = this.executedTrades.filter(t => t.type === 'buy').length;
    const sells = this.executedTrades.filter(t => t.type === 'sell').length;
    const executing = this.executingTrades.size;

    return {
      totalExecuted: this.executedTrades.length,
      buys,
      sells,
      executing,
      successRate: this.executedTrades.length > 0
        ? ((buys / this.executedTrades.length) * 100).toFixed(2)
        : 0
    };
  }
}

// Export singleton instance
const tradeExecutor = new TradeExecutor();
export default tradeExecutor;
