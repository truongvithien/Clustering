# Learner: Truong Vi Thien (14520874)
# Human face datasets loading (for convenience clustering)
# http://scikit-learn.org/stable/datasets/labeled_faces.html

import numpy as np

from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern

# Load data

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


data = np.array([]).reshape(0,1850)
for image in faces.images:
    feature = local_binary_pattern(image, P=8, R=0.5).flatten()
    data = np.append(data,[feature],axis=0)

targets = faces.target
target_names = faces.target_names
n_classes = target_names.shape

np.save(file='data_target.npy', arr=targets)
np.save(file='data_target_name.npy', arr=target_names)
np.save(file='data_feature.npy', arr=data)


# Check
print("Datasets loaded! ")
print("n classes: %d" %n_classes)
print("n samples: %d" %len(faces.images))
