// Dynamically determine API base URL
const API_BASE = window.location.origin;
let currentTab = 'pending';

async function fetchPendingSuggestions() {
  try {
    const url = `${API_BASE}/api/suggestions/pending`;
    console.log('Fetching pending suggestions from:', url);
    const response = await fetch(url);
    
    console.log('Response status:', response.status);
    if (!response.ok) {
      showError('pendingList', `HTTP ${response.status}: ${response.statusText}`);
      return;
    }
    
    const data = await response.json();
    console.log('Received data:', data);
    renderSuggestions(data.suggestions, 'pending');
  } catch (error) {
    console.error('Error fetching suggestions:', error);
    showError('pendingList', `Error: ${error.message}`);
  }
}

async function fetchSuggestionHistory() {
  try {
    const url = `${API_BASE}/api/suggestions/history?limit=50`;
    console.log('Fetching history from:', url);
    const response = await fetch(url);
    
    if (!response.ok) {
      showError('historyList', `HTTP ${response.status}: ${response.statusText}`);
      return;
    }
    
    const data = await response.json();
    console.log('History received:', data);
    renderSuggestions(data.history, 'history');
  } catch (error) {
    console.error('Error fetching history:', error);
    showError('historyList', `Failed to load history: ${error.message}`);
  }
}

async function fetchStatistics() {
  try {
    const url = `${API_BASE}/api/suggestions/stats`;
    console.log('Fetching stats from:', url);
    const response = await fetch(url);
    
    if (!response.ok) {
      showError('statsContent', `HTTP ${response.status}: ${response.statusText}`);
      return;
    }
    
    const data = await response.json();
    console.log('Stats received:', data);
    renderStatistics(data.stats);
    updateStatsGrid(data.stats);
  } catch (error) {
    console.error('Error fetching stats:', error);
    showError('statsContent', `Failed to load statistics: ${error.message}`);
  }
}

function renderSuggestions(suggestions, type) {
  const containerId = type === 'pending' ? 'pendingList' : 'historyList';
  const container = document.getElementById(containerId);

  if (!suggestions || suggestions.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>No ${type} suggestions</h3>
        <p>Trade suggestions will appear here as the bot detects opportunities</p>
      </div>
    `;
    return;
  }

  container.innerHTML = suggestions.map(suggestion => `
    <div class="suggestion-card">
      <div class="suggestion-header">
        <div class="token-info">
          <h3>${suggestion.token.symbol}</h3>
          <div class="token-address">${suggestion.token.address}</div>
        </div>
        <div>
          <div class="confidence-badge ${getConfidenceClass(suggestion.confidence)}">
            ${(suggestion.confidence * 100).toFixed(1)}%
          </div>
          <span class="status status-${suggestion.status}">${suggestion.status}</span>
        </div>
      </div>

      ${suggestion.metrics ? `
        <div class="metrics">
          <div class="metric">
            <div class="metric-label">Volume</div>
            <div class="metric-value">${formatNumber(suggestion.metrics.volume)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Liquidity</div>
            <div class="metric-value">${formatNumber(suggestion.metrics.liquidity)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Holders</div>
            <div class="metric-value">${suggestion.metrics.holders || 0}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Trade Amount</div>
            <div class="metric-value">${suggestion.amount ? suggestion.amount.toFixed(2) : 'N/A'} SOL</div>
          </div>
        </div>
      ` : ''}

      ${suggestion.reasoning && suggestion.reasoning.length > 0 ? `
        <div class="reasoning">
          <div class="reasoning-title">Why This Trade?</div>
          <ul class="reasoning-list">
            ${suggestion.reasoning.map(reason => `<li>${reason}</li>`).join('')}
          </ul>
        </div>
      ` : ''}

      ${suggestion.status === 'pending' ? `
        <div class="actions">
          <button class="btn-accept" data-suggestion-id="${suggestion.id}">Accept Trade</button>
          <button class="btn-reject" data-suggestion-id="${suggestion.id}">Reject</button>
        </div>
      ` : ''}

      <div class="timestamp">
        ${suggestion.status === 'accepted' ? `Accepted: ${new Date(suggestion.acceptedAt).toLocaleString()}` : ''}
        ${suggestion.status === 'rejected' ? `Rejected: ${new Date(suggestion.rejectedAt).toLocaleString()}` : ''}
        ${suggestion.status === 'pending' ? `Expires: ${new Date(suggestion.expiresAt).toLocaleString()}` : ''}
      </div>
    </div>
  `).join('');

  // Attach event listeners to buttons after rendering
  attachSuggestionButtonListeners();
}

function attachSuggestionButtonListeners() {
  document.querySelectorAll('.btn-accept').forEach(btn => {
    btn.addEventListener('click', function() {
      acceptSuggestion(this.dataset.suggestionId);
    });
  });

  document.querySelectorAll('.btn-reject').forEach(btn => {
    btn.addEventListener('click', function() {
      rejectSuggestion(this.dataset.suggestionId);
    });
  });
}

async function acceptSuggestion(suggestionId) {
  try {
    const response = await fetch(`${API_BASE}/api/suggestions/${suggestionId}/accept`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();

    if (data.success) {
      alert(`Trade accepted! ${data.message}`);
      fetchPendingSuggestions();
      fetchStatistics();
    } else {
      alert(`Error: ${data.error}`);
    }
  } catch (error) {
    console.error('Error accepting suggestion:', error);
    alert('Failed to accept suggestion');
  }
}

async function rejectSuggestion(suggestionId) {
  const reason = prompt('Reason for rejection (optional):', '');

  try {
    const response = await fetch(`${API_BASE}/api/suggestions/${suggestionId}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason })
    });
    const data = await response.json();

    if (data.success) {
      alert('Trade rejected');
      fetchPendingSuggestions();
      fetchStatistics();
    } else {
      alert(`Error: ${data.error}`);
    }
  } catch (error) {
    console.error('Error rejecting suggestion:', error);
    alert('Failed to reject suggestion');
  }
}

function renderStatistics(stats) {
  const statsContent = document.getElementById('statsContent');
  statsContent.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
      <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="color: #666; margin-bottom: 10px;">Total Suggestions</div>
        <div style="font-size: 2em; font-weight: bold; color: #667eea;">${stats.totalSuggestions}</div>
      </div>
      <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="color: #666; margin-bottom: 10px;">Accepted</div>
        <div style="font-size: 2em; font-weight: bold; color: #28a745;">${stats.accepted}</div>
      </div>
      <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="color: #666; margin-bottom: 10px;">Rejected</div>
        <div style="font-size: 2em; font-weight: bold; color: #dc3545;">${stats.rejected}</div>
      </div>
      <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="color: #666; margin-bottom: 10px;">Acceptance Rate</div>
        <div style="font-size: 2em; font-weight: bold; color: #667eea;">${stats.acceptanceRate.toFixed(1)}%</div>
      </div>
      <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="color: #666; margin-bottom: 10px;">Avg Confidence</div>
        <div style="font-size: 2em; font-weight: bold; color: #667eea;">${stats.avgConfidence}</div>
      </div>
      <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="color: #666; margin-bottom: 10px;">Pending</div>
        <div style="font-size: 2em; font-weight: bold; color: #ffc107;">${stats.pending}</div>
      </div>
    </div>
  `;
}

function updateStatsGrid(stats) {
  document.getElementById('pendingCount').textContent = stats.pending;
  document.getElementById('acceptedCount').textContent = stats.accepted;
  document.getElementById('rejectedCount').textContent = stats.rejected;
  document.getElementById('acceptanceRate').textContent = stats.acceptanceRate.toFixed(1) + '%';
}

function switchTab(tabName, buttonElement) {
  currentTab = tabName;

  // Remove active class from all tabs
  document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
  
  // Hide all tab content
  document.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');

  // Add active class to clicked tab
  buttonElement.classList.add('active');
  
  // Show the selected tab content
  document.getElementById(tabName + 'Tab').style.display = 'block';

  // Fetch data for the selected tab
  if (tabName === 'pending') fetchPendingSuggestions();
  else if (tabName === 'history') fetchSuggestionHistory();
  else if (tabName === 'stats') fetchStatistics();
}

function getConfidenceClass(confidence) {
  if (confidence >= 0.75) return 'confidence-high';
  if (confidence >= 0.6) return 'confidence-medium';
  return 'confidence-low';
}

function formatNumber(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(2) + 'K';
  return num.toFixed(2);
}

function showError(containerId, message) {
  document.getElementById(containerId).innerHTML = `
    <div class="empty-state">
      <h3>Error</h3>
      <p>${message}</p>
    </div>
  `;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('DOMContentLoaded, API_BASE:', API_BASE);
  console.log('Starting initial data fetch');
  
  // Attach tab listeners
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
      const tabName = this.textContent.toLowerCase().replace('statistics', 'stats');
      switchTab(tabName, this);
    });
  });

  fetchPendingSuggestions();
  fetchStatistics();

  // Auto-refresh every 10 seconds
  setInterval(() => {
    console.log('Auto-refresh triggered, currentTab:', currentTab);
    if (currentTab === 'pending') fetchPendingSuggestions();
    fetchStatistics();
  }, 10000);
});
