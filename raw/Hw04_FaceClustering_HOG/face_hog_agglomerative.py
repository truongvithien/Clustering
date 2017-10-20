# Learner: Truong Vi Thien (14520874)
# Human faces clustering using Agglomerative.

import numpy as np
from time import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load data

data = np.load('data_feature.npy')
target = np.load('data_target.npy')
#np.load('data_target_name.npy')

# Principal components analysis aka Reduce data into 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)

# Using DBSCAN
n_clusters = 7

agglo = AgglomerativeClustering(n_clusters = n_clusters)
agglo.fit_predict(data)

labels = agglo.labels_

# Visualize
print(labels)

plt.title(' Human faces clustering using Agglomerative.')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
plt.show()
