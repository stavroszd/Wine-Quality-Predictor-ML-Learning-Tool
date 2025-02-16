# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:24:52 2025

@author: stavr
"""

#%%Importing the necessary packages 

#Basic Python Packages
import os 
import time
import sys

#Data-Processing
import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt 
#import seaborn as sns 

#PreProcessing
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Pre-Processing custom built functions
from data_preprocessing import np_to_tensor, pd_to_tensor
from data_preprocessing import train_cv_test_split_dataset_and_dataloader
from data_preprocessing import anomaly_detection


#Metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import f1_score

#Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier,plot_tree

#We import the necessary packages for the Neural Network- we will use PyTorch
import torch
from torch import nn
import torch.nn.functional as F

#We make a class of our model - they are in another file and imported at the end
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#We import the necessary packages for the Neural Network- we will use PyTorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#We will first make our Dataset class - this will be useful for the Neural Network
from torch.utils.data import Dataset
from models import WineQualityDataset

#Torch model imports
from models import WinePredictorV0, WineModelOptimized, WineModelBig, Somelier

#Torch utils for training and testing 
from torch_utils import train_step, test_step, NN_train_and_test_once_general, NN_train_test_n_times

#We will use MLflow to log our experiments
import mlflow 
mlflow.set_tracking_uri('http://127.0.0.1:5000')

#Hyperparameter tuning 
import itertools
import optuna
from optuna_utils import objective

#%%Data importing 
df = pd.read_csv('data/winequality-white.csv', delimiter = ';')
#We drop our output 
y = df['quality']
X = df.drop(axis = 'columns', labels = 'quality')
X_in = X.copy()
X_in = X_in.to_numpy() #This will be crucial down the line
#%% Anomaly Detection

#We drop the anomalous data and also get the indexes
X_outliers_indexes, X_in_without_outliers = anomaly_detection(data = X_in, contamination = 0.05, y_included = False)
#We drop them from our y ones as well
y_without_outliers = y.drop(axis = 0, labels = X_outliers_indexes).reset_index(drop = True)

#%%Class Creation for classification
def make_classes(x): 
    if x <=4: return 0 
    elif x == 5 or x == 6: return 1
    else: return 2

y1_without_outliers = y.apply(make_classes) #We don't need the axis argument cause why is a Series 


#%%Typical trial cell

NN_train_test_n_times ( epochs = 250 , n = 1 , X = X_in_without_outliers , y = y_without_outliers, model_class = WineModelOptimized , task = 'regression'
, name = 'Regression' , learning_rate = 0.04950464437244165 ,
weight_decay = 0.00023573619467784986 , batch_size = 32 , input_shape =
11 , output_shape = 1)
#%%