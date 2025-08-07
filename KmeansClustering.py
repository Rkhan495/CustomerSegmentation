# K-MEANS CLUSTERING

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Loading Dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Accessing Rows and Columns
X = dataset.iloc[:, 3:5].values
# X = dataset.iloc[:, [3,4]].values

# Elbow Method
# Initialize an empty list to store the WCSS (Within Cluster Sum of Squares) values for different number of clusters
wcss = []
for i in range(1, 51):  # Iterate over a range of cluster numbers from 1 to 49
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # Create a KMeans instance with 'i' clusters, init='k-means++' is used for better initialization of centroids
    kmeans.fit(X)  # Fit the KMeans model on the dataset X
    wcss.append(kmeans.inertia_)  # Append the inertia (WCSS) of the current model to the wcss list


print(wcss)
plt.plot(range(1, 51), wcss)
plt.show()

# Creates a KMeans instance with 5 clusters (you may choose this based on the elbow plot).
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)
print(y_pred)

# Visualization
plt.scatter(X[y_pred==0, 0], X[y_pred==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_pred==1, 0], X[y_pred==1, 1], s=100, c='green', label='Cluster 2')
plt.scatter(X[y_pred==2, 0], X[y_pred==2, 1], s=100, c='blue', label='Cluster 3')
plt.scatter(X[y_pred==3, 0], X[y_pred==3, 1], s=100, c='yellow', label='Cluster 4')
plt.scatter(X[y_pred==4, 0], X[y_pred==4, 1], s=100, c='cyan', label='Cluster 5')
plt.show()

# Plotting the cluster centers
# Retrieves the coordinates of the cluster centroids
centers = kmeans.cluster_centers_
print(centers)
# Plots the cluster centroids on the same plot with a black 'x' marker.
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='black', label='Centroids', marker='x')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters and Centroids')
# Displays the legend indicating which color corresponds to which cluster and the centroid marker.
plt.legend()
plt.show()