import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# importing machine learning models for prediction
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# loading iris dataset
iris = load_iris()
# getting feature data from the iris dataset
features = iris.data
# getting target data from the iris dataset
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20)
# initializing the base models
model1 = RandomForestClassifier(n_estimators=10, random_state=42)
model2 = SVC(kernel='rbf', probability=True, random_state=42)
model3 = LogisticRegression(max_iter=1000, random_state=42)
# initializing the stacking model
estimators = [('rf', model1), ('svc', model2)]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=model3)
# training the stacking model on the train dataset
stacking_model.fit(X_train, y_train)
# predicting the output on the test dataset
pred_final = stacking_model.predict(X_test)
# printing the accuracy score between real value and predicted value
print(accuracy_score(y_test, pred_final))

# OUTPUT:
# 0.9333333333333333
