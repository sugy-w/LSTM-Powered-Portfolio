import torch
import torch.nn as nn

class DeepPortfolioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_assets):
        super(DeepPortfolioLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Softmax ensures weights sum to 1.0
        self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, 64),      # Intermediate dense layer
                nn.ReLU(),                      # Activation (no dim arguments)
                nn.Linear(64, num_assets),      # Final layer mapping to asset count
                nn.Softmax(dim=-1)              # Ensures weights sum to 1.0
        )

    def forward(self, x):
        # x shape: [batch, time_steps, features]
        lstm_out, _ = self.lstm(x)
        # We only take the last hidden state to decide weights for the next period
        last_hidden = lstm_out[:, -1, :]
        weights = self.output_layer(last_hidden)
        return weights

def sharpe_loss(weights, next_returns, prev_weights=None, gamma=0.01, lambda_tc=0.002):
    """
    gamma: Controls diversification (higher = more stocks)
    lambda_tc: Controls trading costs (e.g., 0.002 = 20 basis points per trade)
    """
    # 1. Calculate Portfolio Return
    port_returns = torch.sum(weights * next_returns, dim=1)
    
    # 2. Transaction Cost Penalty (Change in weights)
    # This keeps the model from 'jittering' or over-trading
    if prev_weights is not None:
        # Penalize the absolute difference between today's and yesterday's weights
        tc_penalty = lambda_tc * torch.sum(torch.abs(weights - prev_weights), dim=1)
        port_returns = port_returns - tc_penalty
    
    # 3. Sharpe Calculation
    expected_return = torch.mean(port_returns)
    risk = torch.std(port_returns) + 1e-6
    sharpe = expected_return / risk
    
    # 4. Diversification (L2 Penalty)
    div_penalty = gamma * torch.norm(weights, p=2)
    
    # We minimize the negative Sharpe + penalties
    return -sharpe + div_penalty