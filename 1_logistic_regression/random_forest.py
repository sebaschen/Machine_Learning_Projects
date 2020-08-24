# -*- coding: utf-8 -*-
"""ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-5Oqk47iUQlulc892_s27ZCKxybzUKfT
"""

from google.colab import files
train_data = files.upload()
test_data = files.upload()

import pandas as pd 
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import io
train = pd.read_csv(io.BytesIO(train_data['train.csv']))
test = pd.read_csv(io.BytesIO(test_data['test.csv']))

sns.heatmap(train.isnull(),yticklabels=False,cbar = False, cmap='viridis')

sns.set_style('whitegrid')

train.head()

train['Age'].plot.hist(bins=36)

import cufflinks as cf

cf.go_offline()

train['Fare'].hist(bins=40,figsize=(10,4))

cf.go_offline()

def impute_age(col):
    Age = col[0]
    Pclass = col[1]

    if pd.isnull(Age):
      if Pclass == 1:
        return 37
      if Pclass == 2:
        return 29
      if Pclass == 3:
        return 24
    else:
      return Age

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

train.drop('Cabin',axis=1,inplace=True)

train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace =True)

X = train.drop(['Survived'],axis=1)
y= train['Survived']#the column we want to predict

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# logmodel = LogisticRegression(max_iter=1000)
# logmodel.fit(X_train,y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
predictions = logmodel.predict(X_test)
confusion_matrix(y_test,predictions)

#cleann the testing data 

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis=1)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace =True)

#training
y = train["Survived"]
features = ["Pclass", "SibSp", "Parch","male","Age","Q","S"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")





