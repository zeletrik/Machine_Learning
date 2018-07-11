# -*- coding: utf-8 -*-

"""
Created on Tue July 11 2018

@author: patrik.zelena
"""
# Chnage working directory
import os
os.chdir('WORKING DIR PATH')

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

path_to_dataset = '50_Startups.csv'
test_size = 0.2

# Importing the dataset
dataset = pd.read_csv(path_to_dataset)

# Last column for Y (dependent variable), may change the index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])

onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into a traning and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
