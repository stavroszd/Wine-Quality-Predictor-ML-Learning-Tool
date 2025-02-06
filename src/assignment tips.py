# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:18:39 2024

@author: stavr
"""
#%%Import Packags

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, train_test_split
from sklearn.decomposition import PCA
from sklearn.models import SVC
from sklearn.metrics import f1_score



#%%Importing Data
df = pd.read_csv('winequality-white.csv', delimiter= ';')

x = df.drop(columns = ['quality'])


#%%Data Preprocessing

#Data Normalization
scaler = StandardScaler()
x_norm = scaler.fit_transform(x)

#Pricinpal Component Analysis to 95% of the initial variance
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_norm)

#%% Make classes for y - either with numpy or pandas

#With numpy - very simple way

y = []

for y_i in df['qualities'].values: 
    if y_i == 3 or y_i == 4: 
        y.append(0)
    elif y_i == 5 or y_i == 6: 
        y.append(1)
    else: y.append(2)
    

y = np.array(y).reshape(-1,1)

#%%Run each model 10 times - !!!!! This is the 10 times part

f1_list_svm = [] #We will keep here the scores of each model 
f1_list_decistion_trees = []

for _ in range(10): 
    
    #Train, test, CV split: 
    x_tr, x_ts, y_tr, y_ts = train_test_split(x_pca,y, test_size=0.2) # get training and testing sets
    x_tr, x_vl, y_tr, y_vl = train_test_split(x_tr,y_tr, test_size=0.1) # get training and validation sets

    #%Make it for each model
        
    #For our SVM 
    model_svm = SVC(decision_function_shape = 'ovo')
    model_svm.fit(x_tr,y_tr)
    y_pred_svm = model_svm.predict(x_ts)
    f1_list_svm.append(f1_score(y_ts, y_pred_svm, average = 'micro'))
     

