# Learner: Truong Vi Thien (14520874)
# Digit data clustering using DBSCAN.

import numpy as np
from time import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn import metrics
from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

# Load data

data = np.load('data_feature.npy')
target = np.load('data_target.npy')
#np.load('data_target_name.npy')

# Principal components analysis aka Reduce data into 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(data)

# Using DBSCAN
neighborhood_distance = 1
min_samples = 1

dbscan = DBSCAN(eps=neighborhood_distance, min_samples=min_samples)
dbscan.fit(data)

labels = dbscan.labels_

# Visualize
print(labels)

plt.title('Digit data clustering using DBSCAN')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
plt.show()
