# Learner: Truong Vi Thien (14520874)
# Human face datasets loading (for convenience clustering)
# http://scikit-learn.org/stable/datasets/labeled_faces.html

import numpy as np

from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern

# Load data

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = faces.images.shape
data = faces.data
n_features = data.shape[1]
targets = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0]

# Check

print("Datasets loaded: ")
print("n samples: %d" %n_samples)
print("n features: %d" %n_features)
print("n classes: %d" %n_classes)

