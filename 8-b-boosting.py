import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# importing machine learning models for prediction
from sklearn.ensemble import GradientBoostingClassifier
# loading iris dataset
iris = load_iris()
# getting feature data from the iris dataset
features = iris.data
# getting target data from the iris dataset
target = iris.target

# Splitting between train data into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20)

# initializing the boosting module with default parameters
model = GradientBoostingClassifier()
# training the model on the train dataset
model.fit(X_train, y_train)
# predicting the output on the test dataset
pred_final = model.predict(X_test)
# printing the accuracy score between real value and predicted value
print(accuracy_score(y_test, pred_final))

# OUTPUT:
# 0.8666666666666667
