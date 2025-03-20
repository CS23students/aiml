import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the data
df = pd.read_csv('data.csv')
# Split the data into training and test sets
X = df.drop('buy_computer', axis=1)
y = df['buy_computer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Train the model
model = GaussianNB()
model.fit(X_train.values, y_train.values)
# Test the model
y_pred = model.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Make a prediction on new data
new_data = np.array([[35, 60000, 1, 100]])
prediction = model.predict(new_data)
print("Prediction:", prediction)


# Sample data.csv file
# age,income,student,credit_rating,buy_computer
# 30,45000,0,10,0
# 32,54000,0,100,0
# 35,61000,1,10,1
# 40,65000,0,50,1
# 45,75000,0,100,0
# OUTPUT:
# Accuracy: 0.0
# Prediction: [1]
