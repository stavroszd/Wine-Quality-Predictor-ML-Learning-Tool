#%%Basic imports
import pandas as pd
import torch

#Necessary sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Models needed 
from sklearn.ensemble import IsolationForest
from models import WineQualityDataset

#Torch imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#%%Makes some tensor processing functions
def np_to_tensor(arrays): 
    return tuple(torch.tensor(data = array, dtype = torch.float32) for array in arrays)

def pd_to_tensor(dfs): 
    return tuple(torch.tensor(data = df.to_numpy(), dtype = torch.long) for df in dfs) #This makes the dataframe a numpy object and then a tensor one

#We will feed the data X and y1 into the decision tree - which will be our 3rd model 
# Importing our packages 

#We make a preprocess function that works the same way
def preprocess_np(X,y): 
    #Do a train - cv - test split  
    X_train, X_cv, X_test, y_train, y_cv, y_test = train_cv_test_split(X,y)
    #Scale the data
    X_train_scaled, X_CV_scaled , X_test_scaled = scale(X_train, X_cv, X_test)

    return X_train_scaled, X_CV_scaled , X_test_scaled, y_train, y_cv, y_test


#%% Anomaly detection function 

def anomaly_detection(data, contamination = 0.05, y_included = False, **kwargs): 
    '''
    Description: Does anomaly detection on the data using isolation forests
    1) Arguments 
    - data: Dataset in pandas 
    - contamination: What percent of the dataset to return as an anomalu
    - *kwargs
    2) Returns 
    - data_without_anomalies: The dataset with the anomalous points removed
    - anomalies_indexes: The indexes that they show up at
    '''

    #Debug: 
    print(pd.__version__)

    #We make sure we are given a pandas dataset 
    
    if not isinstance(data, pd.DataFrame):
        try: 
            data = pd.DataFrame(data)
        except Exception as e: 
            raise TypeError(f'Cannot convert the data to a dataframe')


    #We need to drop the last column first
    X = data.drop(data.columns[-1], axis = 1) if y_included is True else data
    
    #Model
    iso_forest_anomaly = IsolationForest(contamination = contamination) #Make the model

    #1)We get the anomalous points indexes in anomalies_indexes

    #For this we make a dataframe of all the +-1 it produces and concatanate it with data
    anomaly_labels = pd.DataFrame(iso_forest_anomaly.fit_predict(X), columns = ['Anomaly Label']) #This returns an array of labels -1 if outlier
    data_with_anomaly_labels = pd.concat([X, anomaly_labels], axis = 'columns')
    #We now make it into a dataframe with just the anomaly scores and the indexes
    X_with_anomaly_labels = data_with_anomaly_labels.drop( labels = [i for i in X.columns], axis = 'columns')
    anomalies_indexes = X_with_anomaly_labels[anomaly_labels['Anomaly Label'] == -1].index.to_list()
    
    #2)We drop the anomalies
    data_without_anomalies = data.drop(axis = 0, labels = anomalies_indexes).reset_index(drop = True)

    return anomalies_indexes, data_without_anomalies

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

