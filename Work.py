# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:15:27 2019

@author: Benjamin B
"""

# Data Preproccessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# X = dataset with ALL rows from ALL columns(-1)
X = dataset.iloc[:, :-1].values
# Y = dataset with ALL rows from column 3
Y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer # Imputer is a class
# What to look for and How to fix missing data (imputer = object of Imputer class)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Fitting model into area needed
imputer = imputer.fit(X[:, 1:3])
# Applying fix to missing data values in area
X[:, 1:3] = imputer.transform(X[:, 1:3])

"""Encoding string data into numbers to be able to insert 'values' into math 
equations. Categorial data include Country and Purchased data."""
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder # LabelEncoder is a class
# Creating object (labelencoder_X) of class LabelEncoder
labelencoder_X = LabelEncoder()
# Encoding Country column of X into array of numbers
"""Poses an issue since countries are given values and so in the equation
the numbers will be greater but doesn't make sense as countries should not be 
considered higher or lower than one another (could work with small, medium, 
and large in like shrit sizes)"""
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
"""To fix this, rather than having 1 column with each country in it as different
numbers, have 3 columns for the 3 countries and a 1 in the row for that one country.
Ex: Split the different values of the same column into their own respective columns
and give those countries a 1 if correct or 0 if not."""
from sklearn.preprocessing import OneHotEncoder # Need this class to fix
# Create object of the new class and specify categorical feature column
onehotencoder = OneHotEncoder(categorical_features = [0])
# Fit to matrix X (don't need to specify 0 as done that before ^)
X = onehotencoder.fit_transform(X).toarray()
# Make another object of LabelEncoder class for Y
# Simple LabelEncoder will work as this is dependant variable (yes, no)
labelencoder_Y = LabelEncoder()
# Fit and transform Y with labelencoded version
Y = labelencoder_Y.fit_transform(Y)

"""Need to divide some data into training set and test set since you need your 
machine to learn how to adapt to new data rather than memorize and ace the training
set (overfitting). Therefore, need similar results on the train and test sets"""
# Splitting the dataset into the Training set and Test set
# From the sklearn model selection library, import train_test_split
from sklearn.model_selection import train_test_split
# Create the sets (test_size is percent of dataset to put in test set)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
"""Feature scaling is done when values are on much different scales than one
another and so they should be changed to fit on the same scale for the 
mathematical equations to work. EX: Age goes from 30-50 but Salary goes from
40,000 to 90,000. These problems arise since many machine learning models are based
on the Euclidean Distance (distance between 2 points/coordinates) and so since
the Salary values are much much higher, this equation will be dominated by the 
larger values and so Age will become almost irrelevant. To fix this, need to 
Standardize or Normalize the data (slightly different equations)."""
# Import preprocessing library and take standard scaler class
from sklearn.preprocessing import StandardScaler
# Create object of this class
sc_X = StandardScaler()
# Fit and transform training set
X_train = sc_X.fit_transform(X_train)
# Transform X_test (do not need to fit since already fit onto X_train)
X_test = sc_X.transform(X_test)
"""Do we need to fit and transform the dumby variables? It depends* But we will
do it here to practice for X but not Y since Y is already 0 or 1"""




