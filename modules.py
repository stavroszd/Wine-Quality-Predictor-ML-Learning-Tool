# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:19:34 2025

This will have our necessary modules

@author: stavr
"""
#%%
from torch import nn
from torch.utils.data import Dataset

#%%We make our dataset object
class WineQualityDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (array-like): Input features (e.g., NumPy array or PyTorch tensor).
            y (array-like): Labels/targets (e.g., NumPy array or PyTorch tensor).
        """
        self.X = X
        self.y = y

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (X[idx], y[idx]) where X[idx] is the input features, and y[idx] is the label.
        """
        return self.X[idx], self.y[idx]
    
#%%We make different models here

class WineModelOptimized(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units=[256, 128, 64]):
        super(WineModelOptimized, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_shape, hidden_units[0]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(hidden_units[0]),
            nn.Dropout(0.2)
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_units[i], hidden_units[i+1]),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm1d(hidden_units[i+1]),
                nn.Dropout(0.2)
            ))

        self.output_layer = nn.Linear(hidden_units[-1], output_shape)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)  # Standard forward pass
        return self.output_layer(x)

#%%Wine Model Big Architecture
class WineModelBig(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units=[512, 256, 128, 64]):
        super(WineModelBig, self).__init__()

        layers = []

        # Input layer -> First hidden layer
        layers.append(nn.Linear(input_shape, hidden_units[0]))
        layers.append(nn.ReLU())  # ReLU activation
        layers.append(nn.BatchNorm1d(hidden_units[0]))  # Batch Normalization for stable training
        layers.append(nn.Dropout(0.4))  # Dropout for regularization

        # Hidden layers
        for i in range(len(hidden_units)-1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(nn.ReLU())  # ReLU activation
            layers.append(nn.BatchNorm1d(hidden_units[i+1]))  # Batch Normalization
            layers.append(nn.Dropout(0.4))  # Dropout for regularization

        # Output layer
        layers.append(nn.Linear(hidden_units[-1], output_shape))

        # Sequential model that will stack all the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# %% Somelier Model


class Somelier(nn.Module): 
    def __init__(self, input_shape = int, output_shape = int): 
        super(Somelier, self).__init__()

        #We now make our architecture
        self.model = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features = 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), #We add a batch normalization layer --- this first parameter 
            #is the features that should normalize so we pass 64 in
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features = output_shape)
        )

    def forward(self,x): 
        return self.model(x)

# %% Wine Predictor V0

class WinePredictorV0(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(WinePredictorV0, self).__init__()
        
        # Define the architecture using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=16, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.model(x)

# %%
