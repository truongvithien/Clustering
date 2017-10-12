# Learner: Truong Vi Thien (14520874)
# Digit data clustering using Spectral Clustering.

import numpy as np
from time import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import spectral_clustering
from sklearn.datasets import load_digits

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load data

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("Input: \n n_digits = n_cluster: %d, \t n_samples: %d, \t n_features: %d"
      % (n_digits, n_samples, n_features))

print(82 * '_')

# Principal components analysis aka Reduce data into 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)


# Using Spectral Clustering
graph = cosine_similarity(digits.data)
labels = spectral_clustering(graph, n_clusters=n_digits)

# Visualize
print(labels)

plt.title('Digit data clustering using K-Mean')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
plt.show()
