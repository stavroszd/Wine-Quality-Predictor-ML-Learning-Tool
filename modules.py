# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:19:34 2025

This will have our necessary modules

@author: stavr
"""

#%%We make our dataset object

from torch.utils.data import Dataset
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