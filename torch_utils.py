#%%Package Imports

#Basics
import pandas as pd
import numpy as np 
import mlflow

#Torch
import torch
from torch import optim 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#sci-kit learn packages
from sklearn.metrics import f1_score

#We import the necessary packages for the Neural Network- we will use PyTorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#We will first make our Dataset class - this will be useful for the Neural Network
from torch.utils.data import Dataset

#Our own modules
from data_preprocessing import train_cv_test_split_dataset_and_dataloader

#%%Training step
def train_step(data_loader, model, task='classification', loss_fn=None, optimizer_fn=None, learning_rate = 0.01, weight_decay = 0.01):
    # We have to define our loss function and optimizer
    if optimizer_fn is None:
        optimizer_fn = optim.AdamW(model.parameters(), lr= learning_rate, weight_decay = weight_decay) #This should change

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


#%%Testing step 

def test_step(dataloader, model, loss_fn = None, task = 'classification'): 
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


#%%Train and test once function 

def NN_train_and_test_once_general(epochs, X, y, model, task = 'classification', batch_size = 128, name = None, learning_rate = 0.01, weight_decay = 0.01, model_kwargs = {}):
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
    best_cv_loss = float('inf')
    #I will adapt this function to work with the loss instead of the metric - because metrics are most unstable
    best_epoch = 0 
    best_state_dict = None
    
    
    #---------- Training the model -----------------------------------
    #We start an MLflow run 
    with mlflow.start_run(run_name = f'Run for Model: {name}'):
        mlflow.pytorch.autolog() #We autolog the model
        mlflow.log_param('Epoch', epochs) #We log the epochs that we will train on 
        mlflow.log_param('Task', task) #We log the task
        mlflow.log_param('Batch Size', batch_size) #We log the batch size
        mlflow.log_param('Model Name', name) #We log the model name
        mlflow.log_param('Learning Rate', learning_rate) #We log the learning rate

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
                if cv_loss < best_cv_loss:
                    best_cv_loss = cv_loss
                    best_cv_metric = cv_metric
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

#%%Train and test n times function 
def NN_train_test_n_times(epochs, n, X, y, model_class, task='classification', name=None, batch_size=32, learning_rate = 0.01, **kwargs):
    '''
    Arguments: 
    - model_class || The class of the model that we want to use, NOT an instance.
    - model_kwargs || Any extra arguments needed to initialize the model.
    e.g. Batch_size, input_shape, output_shape, learning_rate, weight_decay
    '''

    try: 
        model_kwargs = {'input_shape': kwargs['input_shape'],'output_shape': kwargs['output_shape']}
    except: 
        model_kwargs = {'output_shape': kwargs['output_shape']} #In case we don't give an output shape
    
    metrics = []
    for run in range(n):
        model = model_class(**model_kwargs)  # Create a new instance with the given parameters
        print(f'Run: {run}')
        
        metrics.append(NN_train_and_test_once_general(epochs, X, y, model=model, task=task, name=f'{name}_Run_Number_{run}', batch_size=batch_size, learning_rate = learning_rate))

    return np.mean(metrics)
