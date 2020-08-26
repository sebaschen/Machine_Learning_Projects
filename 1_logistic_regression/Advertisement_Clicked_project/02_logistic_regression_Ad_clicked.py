# -*- coding: utf-8 -*-
# 02-Logistic Regression Project.ipynb


# Original file is located at
# https://colab.research.google.com/drive/1JANVZ9F67iIp4KS7Z-9GAKj8_3XkC5JE




import pandas as pd 
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

## Get the Data

from google.colab import files
train_data = files.upload()
ad_data = pd.read_csv(io.BytesIO(train_data['advertising.csv']))


ad_data.info()
ad_data.describe()

"""## Exploratory Data Analysis

** Create a histogram of the Age**
"""

sns.distplot(ad_data['Age'],kde=False,bins=30,hist_kws={"alpha":0.75})


"""**Create a jointplot showing Area Income versus Age.**
"""

sns.set_style('whitegrid')
sns.jointplot(x="Age",y="Area Income",data=ad_data)

ad_data.columns.values

"""**Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**"""

sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, kind="kde");

"""** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**"""

sns.jointplot(x="Daily Time Spent on Site",y="Daily Internet Usage",data=ad_data)

"""** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**"""

sns.pairplot(ad_data,hue="Clicked on Ad")

"""# Logistic Regression

Now it's time to do a train test split, and train our model!

You'll have the freedom here to choose columns that you want to train on!

** Split the data into training set and testing set using train_test_split**
"""

from sklearn.model_selection import train_test_split
X = ad_data[["Daily Time Spent on Site", 'Age', 'Area Income','Daily Internet Usage', 'Male']]

X_train, X_test, Y_train, Y_test = train_test_split(X,ad_data["Clicked on Ad"],test_size=0.30,random_state=0)

test_data

"""** Train and fit a logistic regression model on the training set.**"""

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train,Y_train)

from sklearn.metrics import classification_report

predictions = logmodel.predict(X_test)
print(classification_report(Y_test,predictions))

"""## Predictions and Evaluations
** Now predict values for the testing data.**
"""

from sklearn.metrics import classification_report

predictions = logmodel.predict(X_test)
print(classification_report(Y_test,predictions))

"""** Create a classification report for the model.**"""





"""## Great Job!"""