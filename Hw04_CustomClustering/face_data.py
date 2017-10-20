# Learner: Truong Vi Thien (14520874)
# Human face datasets loading (for convenience clustering)
# Feature extraction technique: Histogram of Gradient
# http://scikit-learn.org/stable/datasets/labeled_faces.html
# http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html

import numpy as np

from sklearn.datasets import fetch_lfw_people
from skimage.feature import hog

# Load data

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

data = np.array([]).reshape(0,1850)
for image in faces.images:
    feature = hog(image, orientations = 8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
    data = np.append(data,[feature])

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
