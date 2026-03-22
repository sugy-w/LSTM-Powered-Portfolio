from torch.utils.data import Dataset, DataLoader

class PortfolioDataset(Dataset):
    def __init__(self, features, returns, window_size=30):
        """
        features: (N, num_predictors) - e.g., Moving averages, RSI
        returns: (N, num_assets) - Actual asset returns for the next step
        window_size: How many days the LSTM looks back
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        # Subtract window_size so we don't go out of bounds
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        # Extract a slice of features from idx to idx + window_size
        x = self.features[idx : idx + self.window_size]
        
        # The target 'y' is the return on the VERY NEXT day after the window ends
        y = self.returns[idx + self.window_size]
        
        return x, y