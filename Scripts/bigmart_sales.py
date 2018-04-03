# Import important libraries

# Looking at Python Version
import sys
print('Python version ' + sys.version)

# IPython is what you are using now to run the notebook
import IPython
print("IPython version: %6.6s" % IPython.__version__)

# Numpy is a library for working with Arrays
import numpy as np
print( "Numpy version:    %6.6s" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print( "Scipy version:  %6.6s" % sp.__version__)

# Pandas makes working with data frames easier
import pandas as pd
print( "Pandas version:   %6.6s" % pd.__version__)

#sklearn is a library for machine learning algorithms
import sklearn
print( "Sklearn version:  %6.6s" % sklearn.__version__)

#Seaborn is used for create beautiful static plots
import seaborn as sb
print( "Seaborn version:  %6.6s" % sb.__version__)

# Matplotlib is used for creating quick plots
import matplotlib
import matplotlib.pyplot as plt

#urllib allows API calls to websites
import urllib

# time used for timing run times of algorithms
import time

print( "Maplotlib version: %6.6s" % matplotlib.__version__)

import os

location = r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Sales Data\train.csv"
location_test =r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Sales Data\test.csv"
# checks encoding of dataset

from char_encode import char_encode
encoder = char_encode(location)

#loading training & testing data
train = pd.read_csv(location,encoding=encoder)
test = pd.read_csv(location_test, encoding=encoder)

print ("\nTraining Sample:")
print(train.sample(3))

print("Cleaning dataset via Imputation...\n")
p=0

# Clean Outlet Size
# Use the mode Outlet Size for missing values
train_mode = train.Outlet_Size.mode()
test_mode = test.Outlet_Size.mode()

train.Outlet_Size = train.Outlet_Size.fillna(train_mode[0])
test.Outlet_Size = test.Outlet_Size.fillna(test_mode[0])
if train.Outlet_Size.isnull().any()  == False:
	p =+ 1

# Clean Item Weight
# Use the median weight for missing values
train.Item_Weight = train.Item_Weight.fillna(np.nanmedian(train.Item_Weight))
test.Item_Weight = test.Item_Weight.fillna(np.nanmedian(test.Item_Weight))
if train.Item_Weight.isnull().any() == False:
	p =+ 1

# Fix Fat Content variables
train.Item_Fat_Content = train.Item_Fat_Content.replace('LF','Low Fat')
train.Item_Fat_Content = train.Item_Fat_Content.replace('low fat','Low Fat')
train.Item_Fat_Content = train.Item_Fat_Content.replace('reg','Regular')

test.Item_Fat_Content = test.Item_Fat_Content.replace('LF','Low Fat')
test.Item_Fat_Content = test.Item_Fat_Content.replace('low fat','Low Fat')
test.Item_Fat_Content = test.Item_Fat_Content.replace('reg','Regular')

if p == 2:
	print("Missing values imputation successful\n")
else:
	print("Missing values imputation failed\n")
location_3 = r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Sales Data\test_cleaned.csv"
test.to_csv(location_3, index=False)

# Normalizing Item MRP
print('Normalizing the Item_MRP field....\n')
from sklearn import preprocessing
x = train[['Item_MRP']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
train['Normalized'] = pd.DataFrame(x_scaled)
print('Normalization Complete!\n')

# Defining X and y variables and tranforming X values to sklearn friendly format
print('Transforming features and labels....\n')
y = np.array(train.Item_Outlet_Sales)
new_train =train.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis=1)

train_selected = new_train.drop(['Item_MRP','Outlet_Size','Item_Weight','Outlet_Establishment_Year','Outlet_Location_Type','Item_Visibility','Item_Fat_Content'], axis = 1)
X= train_selected.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X = vec.fit_transform(X).toarray()
print('Transformation completed, X shape: {} and Y shape {}\n'.format(X.shape,y.shape) )

#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.04, random_state =247)

print("The data has been split using 96% for training and 4% for testing\n")

# Load all necessary modules needed
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.neural_network import MLPRegressor as NNR
from sklearn.linear_model import BayesianRidge as BR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

print ("Random Forest Regression:")
model= RFR(n_estimators = 5, max_depth = 3)
start_time = time.time()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
elapsed_time = time.time() - start_time
z = sqrt(mean_squared_error(y_test, predicted))
print('RMS Evaluation:  {}'.format(z) )
print('Prediction/Fit Run Time: {}\n'.format(elapsed_time))

print ("Bayesian Ridge:")
model= BR()
start_time = time.time()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
elapsed_time = time.time() - start_time
z = sqrt(mean_squared_error(y_test, predicted))
print('RMS Evaluation:  {}'.format(z) )
print('Prediction/Fit Run Time: {}\n'.format(elapsed_time))

print ("Decision Tree Regression:")
model= DTR(max_depth = 3)
start_time = time.time()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
elapsed_time = time.time() - start_time
z = sqrt(mean_squared_error(y_test, predicted))
print('RMS Evaluation:  {}'.format(z) )
print('Prediction/Fit Run Time: {}\n'.format(elapsed_time))

print ("Linear Regression:")
model= LNR()
start_time = time.time()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
elapsed_time = time.time() - start_time
z = sqrt(mean_squared_error(y_test, predicted))
print('RMS Evaluation:  {}'.format(z) )
print('Prediction/Fit Run Time: {}\n'.format(elapsed_time))

print ("Neutral Network Regression:")
model= NNR()
start_time = time.time()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
elapsed_time = time.time() - start_time
z = sqrt(mean_squared_error(y_test, predicted))
print('RMS Evaluation:  {}'.format(z) )
print('Prediction/Fit Run Time: {}\n'.format(elapsed_time))

print ("Gradient Boosting Regression Score:")
params = {'n_estimators': 50}
model= GBR(**params)
start_time = time.time()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
elapsed_time = time.time() - start_time
z = sqrt(mean_squared_error(y_test, predicted))
print('RMS Evaluation:  {}'.format(z) )
print('Prediction/Fit Run Time: {}\n'.format(elapsed_time))

print('Loading in the test dataset...')
test = pd.read_csv(r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Sales Data\test_cleaned.csv")
print(test.sample(4))

print("\nTransforming data......")
from sklearn import preprocessing
x = test[['Item_MRP']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
test['Normalized'] = pd.DataFrame(x_scaled)

new_test =test.drop(['Item_Identifier','Outlet_Identifier',],axis=1)

test_selected = new_test.drop(['Item_MRP','Outlet_Size','Item_Weight','Outlet_Establishment_Year','Outlet_Location_Type','Item_Visibility','Item_Fat_Content'], axis = 1)
X_2= test_selected.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
test_X = vec.fit_transform(X_2).toarray()

print('\nTest features (X) before: ')
print(test_selected.sample(3))
print('\nTest features (X) after: ')
print(test_X[3])

print('\nPredicting from Testing Data....')
pred_y = model.predict(test_X)
print('\nValues Predicted: ',pred_y)

print('Exporting Predicted data to file...')
test['Item_Outlet_Sales'] = pred_y
test[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']].to_csv(r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Sales Data\predictions.csv", index=False)
print('Exported predicted data to file "predictions.csv"')

from subprocess import Popen
Popen('predictions.csv', shell=True)