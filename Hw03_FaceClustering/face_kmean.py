# Learner: Truong Vi Thien (14520874)
# Human faces clustering using K-Mean.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")


# Load data

np.load('faces_data.npy')
np.load('faces_target.npy')
np.load('faces_target_name.npy')

# Principal components analysis aka Reduce data into 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)


# Using KMean
kmeans = KMeans(n_clusters=n_digits)
kmeans.fit(reduced_data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize
print(centroids)
print(labels)

plt.title('Digit data clustering using K-Mean')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=69, linewidths=5, color='w', edgecolors="k", alpha=0.5, zorder=10)
plt.show()
