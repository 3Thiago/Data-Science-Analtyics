# -*- coding: utf-8 -*-
#Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sb

location =r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Loan Prediction\train.csv"
loan = pd.read_csv(location)
loan.head()

loan['Gender'] = loan['Gender'].fillna('Male')
loan['Married'] = loan['Married'].fillna('Yes')
loan['Dependents'] = loan['Dependents'].fillna(0)
loan['Self_Employed'] = loan['Self_Employed'].fillna('No')
loan['LoanAmount'] = loan['LoanAmount'].fillna(round(loan['LoanAmount'].mean(),1))
loan['Loan_Amount_Term'] = loan['Loan_Amount_Term'].fillna(round(loan['Loan_Amount_Term'].mean(),1))
loan['Credit_History'] = loan['Credit_History'].fillna(round(loan['Credit_History'].mean(),0))

y = np.array(loan.Loan_Status)

loan_selected = loan.drop(['Loan_ID','Loan_Status'], axis = 1)
X= loan_selected.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X = vec.fit_transform(X).toarray()

import random
random.seed(1)

#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1234)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

predicted= model.predict(X_test)
print ("Model accuracy is %.2f" % accuracy_score(predicted, y_test))

location =r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Loan Prediction\test.csv"
loan_test = pd.read_csv(location)

loan_test['Gender'] = loan_test['Gender'].fillna('Male')
loan_test['Married'] = loan_test['Married'].fillna('Yes')
loan_test['Dependents'] = loan_test['Dependents'].fillna(0)
loan_test['Self_Employed'] = loan_test['Self_Employed'].fillna('No')
loan_test['LoanAmount'] = loan_test['LoanAmount'].fillna(round(loan_test['LoanAmount'].mean(),1))
loan_test['Loan_Amount_Term'] = loan_test['Loan_Amount_Term'].fillna(round(loan_test['Loan_Amount_Term'].mean(),1))
loan_test['Credit_History'] = loan_test['Credit_History'].fillna(round(loan_test['Credit_History'].mean(),0))

loan_selected_1 = loan_test.drop(['Loan_ID'], axis = 1)
X_1= loan_selected_1.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
test_x = vec.fit_transform(X_1).toarray()

pred_y= model.predict(test_x)
print("Creating CSV file with prediction results.....")
loan_test['Loan_Status'] = pred_y
loan_test[['Loan_ID', 'Loan_Status']].to_csv(r'C:\Users\Latoya Clarke\Desktop\Data for Analysis\Loan Prediction\results.csv', index=False)
print("File name results.csv is created")
