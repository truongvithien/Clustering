# Learner: Truong Vi Thien (14520874)
# Human faces clustering using K-Mean.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# Load data

data = np.load('data_feature.npy')
target = np.load('data_target.npy')
#np.load('data_target_name.npy')

# Using KMean
n_clusters = 7 # 7 classes in dataset.
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit_predict(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize
print(centroids)
print(labels)

data = PCA(n_components=2).fit_transform(data)

plt.title('Human faces clustering using K-Mean')
plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=69, linewidths=5, color='w', edgecolors="k", alpha=0.5, zorder=10)

plt.show()
