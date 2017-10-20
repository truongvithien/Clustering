# Learner: Truong Vi Thien (14520874)
# Human faces clustering using Spectral Clustering.

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import spectral_clustering

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load data

data = np.load('data_feature.npy')
target = np.load('data_target.npy')
print(data)

# Principal components analysis aka Reduce data into 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)

# Using Spectral Clustering
n_clusters = 7
graph = cosine_similarity(data)
labels = spectral_clustering(graph, n_clusters=n_clusters)

# Visualize
print(labels)

plt.title('Human faces clustering using Spectral Clustering')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
plt.show()
