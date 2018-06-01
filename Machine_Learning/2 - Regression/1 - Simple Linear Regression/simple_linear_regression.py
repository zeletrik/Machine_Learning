# -*- coding: utf-8 -*-
"""
Created on Tue May 29 2018

@author: patrik.zelena
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

path_to_dataset = 'Salary_Data.csv'
test_size = 1/3

# Importing the dataset
dataset = pd.read_csv(path_to_dataset)

# Last column for Y (dependent variable), may change the index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into a traning and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)


# Simple Linear Regression

# Fitting Simple Linear Regression to the Traning set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test set observation
y_pred  = regressor.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary - Experience [Traning set]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary - Experience [Test set]')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
