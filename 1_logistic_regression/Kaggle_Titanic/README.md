# [Titanic: Machine Learning from Disaster Kaggle Challenge](https://www.kaggle.com/c/titanic)


## Summary

It's the classic Titanic Kaggle competition. We're going to use two methodologies to deal with the problem:

1. Logistic Regression: Use logistic regression to find a line that can separate the data to two parts.
2. Random Forest Classifier: Random forest is a versatile machine learning method capable of performing both regression and classification tasks. 
It is also used for dimentionality reduction, treats missing values, outlier values. It is a type of ensemble learning method, where a group of weak models combine to form a powerful model

### Variables &Features of the csv file

| Variable  | Definition | Key |
| ------------- | ------------- |-------------|
| survival  | Survival |0 = No, 1 = Yes |
| pclass  | Ticket class  |1 = 1st, 2 = 2nd, 3 = 3rd |
| sex  | Sex  | |
| sibsp  | spouses aboard the Titanic  ||
| Age  | Age in years  | |
| parch  | # of parents / children aboard the Titanic||
| ticket  | Ticket number | |
| fare  | Passenger fare  | |
| cabin  | Cabin number  ||
| embarked  | Port of Embarkation  |C = Cherbourg, Q = Queenstown, S = Southampton |

## Data Visualization

### Original Data Heatmap
<img src="https://github.com/sebaschen/basic_machine_learning/blob/master/1_logistic_regression/Kaggle_Titanic/original_data_heatmap.png" alt="drawing" width="500"/>

### Age Distribution
<img src="https://github.com/sebaschen/basic_machine_learning/blob/master/1_logistic_regression/Kaggle_Titanic/original_data_age_distribution.png" alt="drawing" width="500"/>


### Ticket Fare
<img src="https://github.com/sebaschen/basic_machine_learning/blob/master/1_logistic_regression/Kaggle_Titanic/Ticket_fare.png" alt="drawing" width="500"/>


### Pclass Rate
<img src="https://github.com/sebaschen/basic_machine_learning/blob/master/1_logistic_regression/Kaggle_Titanic/Pclass_rate.png" alt="drawing" width="500"/>


### Gender
<img src="https://github.com/sebaschen/basic_machine_learning/blob/master/1_logistic_regression/Kaggle_Titanic/sex_survived.png" alt="drawing" width="500"/>

```
Most of the men died, while the women survived.
```

## Result 
| Result:   | Accuracy |
| ------------- | ------------- |
| LogisticRegression  | 0.76555 |
| RandomForestClassifier  | 0.78468 |
