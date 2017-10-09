# Learner: Truong Vi Thien (14520874)
# Generate Gaussian blobs and clustering using K-Mean.

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


# Create data
n_samples = 1000
n_features = 2

x, _ = make_blobs(n_samples=n_samples, n_features=n_features)

# Using KMeans
n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#Visualize
print(centroids)
print(labels)

colors = ["c.","m.","y.","k.","r.","g.","b."]

for i in range(len(x)):
    print("coordinate: ", x[i], "label: ", labels[i])
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize = 5)

plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 150, linewidths= 5, zorder = 10)
plt.show()
