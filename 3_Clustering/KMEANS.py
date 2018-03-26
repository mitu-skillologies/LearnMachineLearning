# K MEANS CLUSTERING

# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Mall Dataset
dataset = pd.read_csv('3_Clustering/Mall_Customers.csv')
# Clients that subscribe to Membership card
# Maintains the Purchase history
# Score is Dependent on INCOME,
#  No. times in week the show up in Mall, total expense in same mall
# YOU ARE!!
# TO Segment Clients into Different Groups based on Income & Score
# CLUSTERING PROBLEM
X = dataset.iloc[:, [3, 4]].values
# We have no Idea to look for
# We don't know the Optimal no. of Clusters

# USE ELBOW METHOD
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    # Fit values into KM
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("ELBOW METHOD")
plt.xlabel("No. of Clus")
plt.ylabel("WCSS")
plt.plot()

# Applying K-means to Mall
kmeans = KMeans(n_clusters=5,  init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the Clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s= 100, c = 'red', label='Cluster1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==0, 1], s= 100, c = 'blue', label='Cluster2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==0, 1], s= 100, c = 'yellow', label='Cluster3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==0, 1], s= 100, c = 'green', label='Cluster4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==0, 1], s= 100, c = 'cyan', label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c = 'red', label='Cluster1')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
