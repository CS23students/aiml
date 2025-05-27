import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# read the data
data = pd.read_csv('flowers.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# create and fit a decision tree model
tree = DecisionTreeClassifier().fit(X_train, y_train)

# visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(tree, filled=True)
plt.title("Decision Tree")
plt.show()

# create and fit a random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

# visualize the random forest (first 6 trees)
plt.figure(figsize=(20, 12))
for i, tree_in_forest in enumerate(rf.estimators_[:6]):
    plt.subplot(2, 3, i + 1)
    plt.axis('off')
    plot_tree(tree_in_forest, filled=True, rounded=True)
    plt.title("Tree " + str(i + 1))

plt.suptitle("Random Forest")
plt.show()

# calculate and print the accuracy of decision tree and random forest
print("Accuracy of decision tree: {:.2f}".format(tree.score(X_test, y_test)))
print("Accuracy of random forest: {:.2f}".format(rf.score(X_test, y_test)))


# Sample flowers.csv
# Sepal_length,Sepal_width,Petal_length,Petal_width,Flower
# 4.6,3.2,1.4,0.2,Rose
# 5.3,3.7,1.5,0.2,Rose
# 5,3.3,1.4,0.2,Rose
# 7,3.2,4.7,1.4,Jasmin
# 6.4,3.2,4.5,1.5,Jasmin
# 7.1,3,5.9,2.1,Lotus
# 6.3,2.9,5.6,1.8,Lotus
