# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:41:49 2018

@author: Anirudh
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
print('################################# Linear regression on Iris Dataset #############################')
############## Load Iris dataset ##############
filename = "irisDataset.txt"
irisDataset = pd.read_csv(filename, sep=',')
############ Data Preprocessing ###############
mapping = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
irisDataset.category = [mapping[i] for i in irisDataset.category]
X_data = np.array(irisDataset.iloc[:,0:4])
Y_data = np.array(irisDataset.iloc[:,4])
beta = [[]]
err_arr = []
############# Perform K Fold Cross validation ############
K = 5 # Change K value to do N fold cross validation 
j = 0
split = int(len(X_data)/K)
t = split
for i in range(0,K):
################ Split data into test and train according to K fold where there is one test set and K -1 training sets in each fold #######    
    rem = []
    X_test = [[]]
    y_test = []
    X_train = [[]]
    y_train = []
    X_test = X_data[j:t]
    y_test = Y_data[j:t]
    rem = list(range(j,t))
    X_train = np.delete(X_data,rem,axis = 0)
    y_train = np.delete(Y_data,rem,axis = 0)
############# Applying linear regression formula beta = (inv(X.T dot X)) dot X.T dot Y  (Least square error method)  
    t1 = inv(np.matmul(X_train.T,X_train))
    t2 = np.matmul(t1,X_train.T)
    t3 = np.matmul(t2,y_train)
    beta.append((t3))
########## Calculate the predict label using Y = X_test dot beta #############
    pred = np.matmul(X_test,t3)
    pred = np.round(pred)
######### Calculate the error of the linear model ########################3
    err = pred - y_test
    err_cnt = np.count_nonzero(err)
    err_arr.append(err_cnt/len(err))
    print ('-----------------------')
    print('predict:',pred)
    print('Truth labels',y_test)
    print('------------------------')
    j = t
    t +=split
err_percent = np.mean(err_arr)*100
#################### Print error and accuracy ########################
print('The error is',err_percent,'%')
accuracy = 100 - err_percent
print('The accuracy is' ,accuracy,'%')