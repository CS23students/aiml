import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# importing machine learning models for prediction
import xgboost as xgb
# importing bagging module
from sklearn.ensemble import BaggingRegressor
# loading Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# getting target data from the dataset
target = iris.target
# getting train data from the dataset
train = pd.DataFrame(iris.data, columns=iris.feature_names)
# Splitting between train data into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(
train, target, test_size=0.20)
# initializing the bagging model using XGBoost as base model with default parameters
model = BaggingRegressor(estimator=xgb.XGBRegressor())
# training model
model.fit(X_train, y_train)
# predicting the output on the test dataset
pred = model.predict(X_test)
# printing the mean squared error between real value and predicted value
print(mean_squared_error(y_test, pred))

# OUTPUT:
# 0.11821985274177556
