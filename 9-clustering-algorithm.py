from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()
# Extract the data and target values
X = iris.data
y = iris.target
# Create a KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10)
# Fit the KMeans object to the data
kmeans.fit(X)
# Get the predicted cluster labels
labels = kmeans.predict(X)
# Plot the data points and centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200,
c='red')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
