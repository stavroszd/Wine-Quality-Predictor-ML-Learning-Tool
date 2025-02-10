# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:24:52 2025

@author: stavr
"""

#%%Data Preprocessing

#%%Importing the necessary packages 


#Data-Processing
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

#PreProcessing
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Model Imports
from sklearn.linear_model import LogisticRegression

#%%Data importing 
df = pd.read_csv('C:/Users/stavr/Desktop/5ο εξάμηνο/statistical learning/Assignment/data/winequality-white.csv', delimiter = ';')
#We drop our output 
y = df['quality']
X = df.drop(axis = 'columns', labels = 'quality')
X_in = X.copy()
X_in = X_in.to_numpy() #This will be crucial down the line

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   
df_scaled = pd.DataFrame(X_scaled, columns = X.columns) 

#%%EDA - Exploratory Data Analysis
#This will be useful if we do some stuff down the line for the pipeline 
#This probably will need to be re-written given our new knowledge - that's why I deleted it

#%%--- Part 3 of assignment --- Anomaly Detection -------

#We will use an isolation forest algorithm to detect anomalies in the data

#Notes: 
#1) This algorithm essentially makes a decision tree and then it isolates the anomalies in the data
#2) The anomalies are isolated quicker because of the split in the decision tree and become leaves way quicker
#3) For more stability and better predictions we do tree ensembles instead of one tree

#Let's make the model object --------
from sklearn.ensemble import IsolationForest
iso_forest_anomaly = IsolationForest(contamination = 0.05) #We use 0.05 for contamination
#because we are not expecting a very big number of anomalies - may adjust later

#Fit the data
iso_forest_anomaly.fit(X_scaled) #We use scaled features?
anomaly_labels = pd.DataFrame(iso_forest_anomaly.fit_predict(X_scaled), columns = ['Anomaly Label']) #This returns an array of labels -1 if outlier

#Let's make an outlier dataframe 
X_outlier = pd.concat([X, anomaly_labels], axis = 'columns')
X_outlier.drop( labels = [i for i in X.columns], axis = 'columns', inplace = True)
X_outliers_indexes = X_outlier[ X_outlier['Anomaly Label'] == -1].index.to_list()


#%%Train - test - split 
#Will try to remove the seed 

def train_cv_test_split(X,y): 
    #We make our train - temp data split first in order to further split temp into Cv and test
    X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size = 0.3)
    X_cv , X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 2/3)
    
    return X_train, X_cv, X_test, y_train, y_cv, y_test

#%%Feature Scaling in Train - CV - Test

def scale(X_train, X_CV, X_test ,y = None): 
    #We will make a function that does the scaling and if you need to scale the outputs 
    #e.g. Regression Model then use a list of the form y_train, ,y_cv, y_test
    
    #You can use a validation set or not if you want
    
    scale_obj = StandardScaler()
    
    #Fit it only on the training set
    X_train_scaled = scale_obj.fit_transform(X_train)
    
    #Then apply that same m,sigma to the others 
    
    #For the input 
    try: X_CV_scaled = scale_obj.transform(X_CV)
    except: print('You need to provide a validation set')
        
    try: X_test_scaled = scale_obj.transform(X_test)
    except: print('You need to provide a test set')

    #For the output ----- I am not going to do this right now
    
    return X_train_scaled, X_CV_scaled , X_test_scaled
    


#%%  Part 1 --- Classification
#This code is poor and needs to be re-written but I can't do it now because I need to keep the pipeline going

# We need to do some further data preprocessing here to turn the quality from numbers to categories 
y1 = y.copy() #We make a y1 for classification copy

# With boolean masking 
y1.loc[y1 <= 4] = 0 #Bad 
y1.loc[(y1 == 5) | (y1 == 6)] = 1 #Medium
y1.loc[y1 > 6 ] = 2 #Good

#%%Anomaly Detection - We will remove the outliers from the data
#Probably we will have to re-write this code as well - would be more useful to do it at the start of the pipeline 
#after the anomaly detection

#We will remove the outliers from the data
X_in_without_outliers = df_scaled.drop(axis = 0, labels = X_outliers_indexes).reset_index(drop = True).to_numpy()
y1_without_outliers = y1.drop(axis = 0, labels = X_outliers_indexes).reset_index(drop = True)

#%% Model 1: Logistic Regression 


#I will make a function that will do all of this and I will just loop it    
def logistic_regression_once(): 
    
    #We split our data - we use y1 because we want to do it with our 0,1,2 format data
    #We use X instead of X_scaled because we want to do the scaling only on the training data
    X_train, X_cv, X_test, y_train, y_cv, y_test = train_cv_test_split(X, y1)
    
    #Proper standard scaling 
    X_train, X_cv, X_test = scale(X_train, X_cv, X_test)
    
    #We make our model
    model = LogisticRegression()
    model.fit(X_train, y_train) #Fit only the training data
    y_preds = model.predict(X_train)
        
    f1 = f1_score(y_train, y_preds, average = 'micro')
    parameters = model.coef_
    
    return f1, parameters


#Now we will do it 10 times and get the mean f1 score

def logistic_10_times(): 
    scores = np.zeros(10)
    parameters = []
    for i in range(10):
        scores[i] = logistic_regression_once()[0]
        parameters.append(logistic_regression_once()[1])
    
    #We calculate and print the mean f1 score
    mean_score = np.mean(scores)
    #We calculate and print the mean of each parameter
    #We add all the parameter arrays and then divide them by 10 to get the mean of each 
    final_array = np.zeros(shape = parameters[0].shape)
    #For better generalization shape could be shape = parameters[0].shape
    for i in range(10):
        final_array += parameters[i]
        final_array = final_array / 10
        
    print(f'The mean f1 of the 10 iterations is {mean_score}')

    #We turn this into a data-frame for easier reading 
    final_array = pd.DataFrame(final_array, columns = X.columns)
    print(f'The mean of the parameters is {final_array}')

    return mean_score, final_array


#%% We now run the model and get the score and parameters
logistic_f1, logistic_params = logistic_10_times()

#We will now find the top 3 values in each row which is what we want 

top_3_features_logistic = logistic_params.apply(lambda row: row.nlargest(3).to_dict(), axis = 1)
#We will make this into a data frame and then we are done

#%%Model 2: Neural Network 

#We import the necessary packages - we will use PyTorch

import torch
from torch import nn

#We will create our architecture
import torch.nn.functional as F

#We make a class of our model
import torch
import torch.nn as nn
import torch.nn.functional as F

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




#%%PyTorch workflow 

#---------- I need to debug this ----------------


#Now let's instantiate our model - this first time around with all 11 features and 3 outputs
#because we want a Classification one 
NNmodelV0 = WinePredictorV0(input_shape = 11, output_shape = 3)


#We will utilise the Dataset and DataLoader utils because they make it more efficient through batching and shuffling

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#Utils function creation

#We will first make our Dataset class

from torch.utils.data import Dataset
from modules import WineQualityDataset

#%%
def np_to_tensor(arrays): 
    return tuple(torch.tensor(data = array, dtype = torch.float32) for array in arrays)

def pd_to_tensor(dfs): 
    return tuple(torch.tensor(data = df.to_numpy(), dtype = torch.long) for df in dfs) #This makes the dataframe a numpy object and then a tensor one

#%%

#We will make a couple of more helper functions now for our training
def train_cv_test_split_dataset_and_dataloader(X,y, batch_size = 32):

  '''This function will 
  1) Take the data and do a split 
  2) Scale like we want to 
  3) Return it as WineQuality dataset objects and as dataloader'''

  #Do a train - cv - test split  
  X_train, X_cv, X_test, y_train, y_cv, y_test = train_cv_test_split(X,y)
  #Scale the data
  X_train_scaled, X_CV_scaled , X_test_scaled = scale(X_train, X_cv, X_test) 
  #Turn them into tensors
  X_train_scaled, X_CV_scaled , X_test_scaled = np_to_tensor((X_train_scaled, X_CV_scaled , X_test_scaled))
  y_train, y_cv, y_test = pd_to_tensor((y_train, y_cv, y_test))  

  #I wanna turn them into tensors

  # Ensure that the lengths of X and y are the same
  assert len(X_train_scaled) == len(y_train), "Mismatch in length of X_train_scaled and y_train"
  assert len(X_CV_scaled) == len(y_cv), "Mismatch in length of X_CV_scaled and y_cv"
  assert len(X_test_scaled) == len(y_test), "Mismatch in length of X_test_scaled and y_test"
 

  #Creating the DataSet objects 
  train_dataset = WineQualityDataset(X_train_scaled, y_train)
  cv_dataset = WineQualityDataset(X_CV_scaled, y_cv)
  test_dataset = WineQualityDataset(X_test_scaled, y_test)
  datasets = [train_dataset, cv_dataset, test_dataset]

  #Creating the dataloaders from them 

  #Note that we need the train one to shuffle as this will make a better fit
  train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = batch_size, num_workers=0) #We do not use num_workers because we are on windows
  #------- This is very important - DEBUG - I think we have to let shuffle = False because 
  #our train test split is already down in random seed so this might explain for the randomness in the key error!
  cv_dataloader = DataLoader(cv_dataset, batch_size=batch_size, shuffle=False, num_workers = 0 ) #We do not shuffle the test and cv data
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 0) #We do not shuffle the test and cv data
  dataloaders = [train_dataloader, cv_dataloader, test_dataloader]

  dataset_dataloader_dict = {'datasets': datasets, 'dataloaders': dataloaders}
                            

  return dataset_dataloader_dict



#%%We will make our training loop finally
from torch import optim

def train_step(data_loader, model=NNmodelV0, task='classification', loss_fn=None, optimizer_fn=None):
    # We have to define our loss function and optimizer
    if optimizer_fn is None:
        optimizer_fn = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) #This should change

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    # We tell it - use CrossEntropyLoss if you are doing classification, otherwise use MSELoss for regression

    # No1 - Set the model to training mode
    model.train()

    # We want to track the loss to be able to assess it later
    train_loss = 0

    # No2 - Do the forward pass
    for batch, (X, y) in enumerate(data_loader):  # For each batch in the train data loader
        #----- try:
        #    print(f"Processing batch {batch} with indices {data_loader.batch_sampler.sampler.data_source.indices[batch * data_loader.batch_size:(batch + 1) * data_loader.batch_size]}")
        #except AttributeError:
        #    print(f"Processing batch {batch}")
        
        y_pred = model(X)  # Forward pass

        # No 3 -- Compute loss and make predictions
        if task == "classification":
            loss = loss_fn(y_pred, y.long())  # CrossEntropyLoss expects y as long type for classification
            y_pred_labels = y_pred.argmax(dim=1)  # Get predicted labels from logits
        else:
            y_pred = y_pred.squeeze()  # Squeeze to remove extra dimension for regression
            loss = loss_fn(y_pred, y.float())  # Use MSELoss for regression, targets should be float

        # No4 - Optimizer zero grad
        optimizer_fn.zero_grad()

        # No 5 - Loss backward
        loss.backward()

        # No 6 - Optimizer update
        optimizer_fn.step()

        # No 7 - Track the loss 
        train_loss += loss.item()  # Add batch loss to total train loss

    # Finally, we want to track the mean batch loss
    avg_loss = train_loss / len(data_loader)  # Mean train loss across batches

    # Metric computation
    if task == 'classification':
        metric = f1_score(y, y_pred_labels, average='micro')  # F1 score for classification
    else:
        metric = torch.sqrt(torch.tensor(avg_loss)).item()  # RMSE for regression

    print(f"Train Loss: {avg_loss:.4f} | Train Metric ({'F1' if task == 'classification' else 'RMSE'}): {metric:.4f}")

    return avg_loss, metric





#%%We will also make our testing step

#We probably can use 
def test_step(dataloader, model = NNmodelV0, loss_fn = None, task = 'classification'): 
  '''
  We need to import the test_dataloader here
  '''
  test_loss, f1_acc = 0, 0 #We set our test_loss and f1_acc to zero because we wanna measure it 
  all_preds, all_labels = [], [] #We make these two arrays that will hold what we get - we will use that for the F1
  
  model.eval() #Put the model on evaluation mode

  with torch.inference_mode(): #We use inference mode because of the same reason - it is a context manager
    for X,y in dataloader: #Loop through the data loader - using batches
      #Make a forward pass
      test_pred = model(X) 
      
      #We calculate the loss - depending on the task - same as the other function
      if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()

      loss = loss_fn(test_pred, y) 
      test_loss += loss.item() 

      #Make inference 
      # No 3 -- Compute loss and make predictions
      if task == "classification":
            loss = loss_fn(test_pred, y.long())  # CrossEntropyLoss expects y as long type for classification
            test_pred_labels = test_pred.argmax(dim = 1).numpy()  # Get predicted labels from logits
            #Store all the labels and predictions for classification - we want it as numpy arrays to then on compute the f1-score
            all_preds.extend(test_pred_labels)
            all_labels.extend(y.numpy())
      else:   
            test_pred = test_pred.squeeze()  # Squeeze to remove extra dimension for regression
            loss = loss_fn(test_pred, y.float())  # Use MSELoss for regression, targets should be float


    # Metric computation

    #Our loss is the average loss everywhere
    avg_loss = test_loss / len(dataloader)

    if task == 'classification':
        metric = f1_score(y, test_pred_labels, average='micro')  # F1 score for classification
    else:
        metric = torch.sqrt(torch.tensor(avg_loss)).item()  # RMSE for regression    

    print(f"Test Loss: {avg_loss:.4f} | Test Metric ({'F1' if task == 'classification' else 'RMSE'}): {metric:.4f}")

    return test_loss, metric #We will use the f1_acc later on

#%%------------- Model 3: Decision Tree ---------------------------------------------------

#We will feed the data X and y1 into the decision tree - which will be our 3rd model 
# Importing our packages 

#We make a preprocess function that works the same way
def preprocess_np(X,y): 
    #Do a train - cv - test split  
    X_train, X_cv, X_test, y_train, y_cv, y_test = train_cv_test_split(X,y)
    #Scale the data
    X_train_scaled, X_CV_scaled , X_test_scaled = scale(X_train, X_cv, X_test)

    return X_train_scaled, X_CV_scaled , X_test_scaled, y_train, y_cv, y_test


#%% Making the model
#Importing the necessary packages 
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -- We instantiate our model --

#Hyperparameters --------------------------------------------------------
#We will use the gini impurity criterion - just like we learned in class
#We will use max_depth = 5 for now to start - we can always expirmenet with this
wine_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 5)

#%%

#We will make a function that trains and tests our tree

def tree_train_test(): 
    #We preprocess our data
    X_train, X_cv, X_test, y_train, y_cv, y_test = preprocess_np(X_in, y1)
    #We will train our model 
    wine_tree.fit(X_train, y_train)
    #We will test our model 
    y_preds = wine_tree.predict(X_test)
    #We will get the f1 score 
    f1 = f1_score(y_test, y_preds, average = 'micro')

    return f1 

#%%We will make a function that does it 10 times 

def tree_10_times():
    f1s = []
    for i in range(10): f1s.append(tree_train_test())

    return np.mean(f1s)
#%%--------------------------------------- Part 2 ------------------------------------------------

#This is our regression part of the assignment and we will make predictions on the quality of the wines
#Since we will need to do Regression we will import our RMSE function 
from sklearn.metrics import mean_squared_error as MSE


#%% ---------- Model 1: Linear Regression --------------------------------------------------------
from sklearn.linear_model import Ridge

#We start of with a simple as possible model - a linear regression model
#The procedure is as simple as it gets. 


def train_linear_regression_once(): 
    #We will make a Linear Regression model using regularization
    ridge_model = Ridge(alpha = 0.001) #We start with a small regularization parameter like we learned in class

    #We will pre-process our data 
    X_train, X_cv, X_test, y_train, y_cv, y_test = preprocess_np(X_in, y)
    #Note we pass in y and not y since we have to predict a continious variable. 

    #We will train our model
    ridge_model.fit(X_train, y_train)
    #We will make predictions
    y_preds = ridge_model.predict(X_test)

    #We will compute our RMSE for the true labels and the predicted labels
    rmse = np.sqrt(MSE(y_test, y_preds))

    return rmse


#%%
def linear_regression_10_times(): 
    rsmes = []
    for i in range(10): rsmes.append(train_linear_regression_once())

    return np.mean(rsmes)

#%%---We will try the same code but with an Elastic Net model

from sklearn.linear_model import ElasticNet
def ElasticNet_once(): 
    #We will make a Linear Regression model using regularization
    ridge_model = ElasticNet(alpha = 0.01, l1_ratio = 1) #We start with a small regularization parameter like we learned in class

    #We will pre-process our data 
    X_train, X_cv, X_test, y_train, y_cv, y_test = preprocess_np(X_in, y)
    #Note we pass in y and not y since we have to predict a continious variable. 

    #We will train our model
    ridge_model.fit(X_train, y_train)
    #We will make predictions
    y_preds = ridge_model.predict(X_test)

    #We will compute our RMSE for the true labels and the predicted labels
    rmse = np.sqrt(MSE(y_test, y_preds))

    return rmse

def ElasticNet_10_times(): 
    rsmes = []
    for i in range(10): rsmes.append(ElasticNet_once())
    print(f'The mean RMSE of the 10 iterations is {np.mean(rsmes)}')
    print(f'The minimum RMSE of the 10 iterations is {np.min(rsmes)}')
    print(f'The maximum RMSE of the 10 iterations is {np.max(rsmes)}')
    print(f'The standard deviation of the RMSE of the 10 iterations is {np.std(rsmes)}')
    return 


#%%
#---------------- Model 2: Neural Network --------------------------------------------------------

#We will make a new model for the regression part of the assignment

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
    
#Let's make an instance of our model 

SomelierV0 = Somelier(input_shape = 11, output_shape = 1)
    
#%% We wanna introduce MLflow logging 
import mlflow 
mlflow.set_tracking_uri('http://127.0.0.1:5000')

#%% This package helps us deal with directories and files 
import os
import time

#%% We will now train and test our model 

#We will make a function that trains and tests our model
#This time we will make it a bit more general

def NN_train_and_test_once_general(epochs, X, y, model, task = 'classification', batch_size = 128, name = None):
    METRIC_NAME = 'F1' if task == 'classification' else 'RMSE' #This will be helpful in printing
    #The path can be modified to adopt to the directory that it needs to be saved to
    model_weights_path = f'results/{name}_best_weights.pth' if name is not None else 'results/best_weights.pth'#This will be the path where we save the weights of the model

    #Data Preprocessing - using our premade function
    dictionary = train_cv_test_split_dataset_and_dataloader(X,y, batch_size = batch_size)
    train_dataloader = dictionary['dataloaders'][0]
    cv_dataloader = dictionary['dataloaders'][1]
    test_dataloader = dictionary['dataloaders'][2]
    
    #We will make some matrices to keep track of the metrics
    #This will be useful in order to keep the model parameters that give the best CV los
    train_metrics = []
    cv_metrics = []
    
    #We set the best metric value to the lowest possible value for classification and the highest possible value for regression
    #This happens because in one the objective is to find the highest value and in the other the lowest
    best_metric_value = float('-inf') if task == 'classification' else float('inf') 
    best_epoch = 0 
    best_state_dict = None
    
    
    #---------- Training the model -----------------------------------
    #We start an MLflow run 
    with mlflow.start_run(run_name = f'Run for Model: {name}'):
        mlflow.log_param('Epoch', epochs) #We log the epochs that we will train on 
        mlflow.log_param('Task', task) #We log the task
    
        #We start our training loop 
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            #------- Training step -------------------
            #We make the training step obtaining the train loss and metric
            train_loss, train_metric = train_step(train_dataloader, model = model, task = task) #The other defaults suffice
            #We log the train cost and metric 
            mlflow.log_metric('Train Cost', train_loss, step = epoch)
            mlflow.log_metric(f'Train {METRIC_NAME}', train_metric, step = epoch)
            
            #Monitor the Validation Cost - every 10 epochs 
            if epoch % 10 == 0: 
                #We get the CV loss and metric 
                cv_loss, cv_metric = test_step(cv_dataloader, model=model, task=task)
                #We log the CV cost and metric
                mlflow.log_metric('CV Cost', cv_loss, step = epoch)
                mlflow.log_metric(f'CV {METRIC_NAME}', cv_metric, step = epoch)
                #We get the train and cv metrics to be able to plot them later
                train_metrics.append(train_metric)
                cv_metrics.append(cv_metric)
            
                #We keep track of the best model parameters
                #This would mean updating the best metric value and the best epoch
                if (task == 'classification' and cv_metric > best_metric_value) or (task == 'regression' and cv_metric < best_metric_value): 
                    best_metric_value = cv_metric
                    best_epoch = epoch
                    best_state_dict = model.state_dict()
            
        #We save the best models weights as an artifact
        if best_state_dict is not None: 
            #Save the best_model weights to a file
            torch.save(best_state_dict, model_weights_path)

            #Log these best weights as an artifact
            mlflow.log_artifact(model_weights_path)
            mlflow.log_param('Best State Dict Saved', True) #Just to make sure that it is done - it will show up in logging

            #We also log the model itself
            mlflow.pytorch.log_model(model, "Best Model")

        #------------------ Testing step -------------------------
        
        #We load the best model weights into the model and do the testing
        if best_state_dict is not None: model.load_state_dict(best_state_dict)

        #Now that the model has been initialized with the best weights we do the test step
        test_loss, test_metric = test_step(test_dataloader, model=model, task=task)
        mlflow.log_metric('Test Cost', test_loss)
        mlflow.log_metric(f'Test {METRIC_NAME}', test_metric)

    return test_metric

#%%Debug cell 



#%%
#We make this function that will train and test n times
#returning the mean of the metrics which is good practice
#The rest will be logged in MLflow and we can pull it from there whenever we want it
def NN_train_test_n_times(epochs, n, X, y, model_class, task='classification', name=None, batch_size=32, **model_kwargs):
    '''
    Arguments: 
    - model_class || The class of the model that we want to use, NOT an instance.
    - model_kwargs || Any extra arguments needed to initialize the model.
    '''
    
    metrics = []
    for run in range(n):
        model = model_class(**model_kwargs)  # Create a new instance with the given parameters
        print(f'Run: {run}')
        
        metrics.append(NN_train_and_test_once_general(epochs, X, y, model=model, task=task, name=f'{name}_Run_Number_{run}', batch_size=batch_size))

    return np.mean(metrics)

#%% Going to try a new model as well

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
# %% We make an instance
NNModelV1_Big = WineModelBig(input_shape = 11, output_shape = 3)

#%% -------------------------------------------- Final Pipeline --------------------------------------------
''''
We will now make a pipeline that will do the following:
    - Train and test the classification Neural Network model 10 times and record it in ML flow 
    - Train and test the regression Neural Network model 10 times and record it in ML flow
'''


#%%------------------ Part 1: Classification ----------------------------------------------------------------
'''
We provide
- epochs: 200 because we saw there was convergence by then for sure
- n: 10 because we want to do it 10 times
- X: X_in because we want to use the data with the anomalies deleted
- y1: y1 because we want to use the data subdivided into classes
- model: WineModelBig because we want to use the big model
- name: 'Wine Classifier' because we want to name the model in MLflow
'''

#We reset the models parameters
NN_train_test_n_times(epochs = 250, n = 10, X = X_in, y = y1, model_class = WineModelBig, task = 'classification', name = 'Wine Classifier', batch_size = 128, input_shape=11, output_shape=3)

#Note that the ones at the end are kwargs that are passed to the model class to instaniate it with proper parameters

#We 
# %% Debug Cell 

#Let me run one by each self


NN_train_and_test_once_general(epochs = 500, X = X_in, y = y1, model = NNModelV1_Big, task = 'classification', name = 'Wine Classifier', batch_size = 128)
# %%
