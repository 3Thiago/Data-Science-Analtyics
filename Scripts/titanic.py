import numpy as np #For arrays and mathematical computations
import pandas as pd # For the dataframe management and manipulation
import sklearn as sk # For utilizating machine learning algorithms

location = r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Titanic Data\train.csv"
titanic = pd.read_csv(location)

titanic['Age']= titanic['Age'].fillna(int(titanic['Age'].mean()))
titanic['Embarked'] = titanic['Embarked'].fillna('S')

#Creating labels for the dataset
y = np.array(titanic.Survived)

#Creating features for training
titanic_selected = titanic.drop(['PassengerId','Survived','Cabin', 'Ticket', 'SibSp','Name'], axis = 1)
X= titanic_selected.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X = vec.fit_transform(X).toarray()

#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

#The data has been split using 80% for training and 20% for testing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


model= RandomForestClassifier(n_estimators= 100)

# Train the model using the training sets and check score
model.fit(X_train, y_train)
score =model.score(X_train, y_train)

#Predict Output
predicted= model.predict(X_test)

print("{0} / {1} correct".format(np.sum(y_test == predicted), len(y_test)))
print("Random Forest Model Accuracy: ", round(100* accuracy_score(predicted, y_test),2),"%")

location = r"C:\Users\Latoya Clarke\Desktop\Data for Analysis\Titanic Data\test.csv"
titanic_test = pd.read_csv(location)

titanic_test['Fare']= titanic_test['Fare'].fillna(int(titanic_test['Fare'].mean()))
titanic_test['Age']= titanic_test['Age'].fillna(int(titanic_test['Age'].mean()))
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')

titanic_selected_1 = titanic_test.drop(['PassengerId','Cabin', 'Ticket', 'SibSp','Name'], axis = 1)
X_1= titanic_selected_1.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
test_x = vec.fit_transform(X_1).toarray()

pred_y = model.predict(test_x)
titanic_test['Survived'] = pred_y
titanic_test[['Name', 'Survived']].to_csv(r'C:\Users\Latoya Clarke\Desktop\Data for Analysis\Titanic Data\resultrf.csv', index=False)
