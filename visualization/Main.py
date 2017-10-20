"""
    Learner: Truong Vi Thien (14520874)
             thientv.cs@gmail.com
    Content: Clustering Algorithms
    Datasets:
        - Gaussian blobs
        - Handwritten digits (http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)
        - Labeled human faces wild (http://scikit-learn.org/stable/datasets/labeled_faces.html) - 
    Techniques: 
        - Feature Extraction: 
            + Local Binary Pattern ()
            + Histogram of Gradient (http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) 
        - Clustering:
            + K-Mean
            + DBSCAN
            + Spectral Clustering
            + Agglomerative Clustering
    
"""""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from time import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn.datasets   import samples_generator, load_digits, fetch_lfw_people
from sklearn.decomposition import PCA

from skimage.feature    import local_binary_pattern, hog
from sklearn.cluster    import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering

#------------------------------------------------------------------------------------------------

data_gaussian = []
data_digits = []
data_lbp_face = []
data_hog_face = []

#------------------------------------------------------------------------------------------------

def init_blobs(n_samples, n_features):
    n_samples = 1000
    n_features = 2
    data, _ = samples_generator.make_blobs(n_samples=n_samples, n_features=n_features)
    return data

def init_digits():
    digits = load_digits()
    data = digits.data
    truth_labels = digits.target
    return data, truth_labels

def init_faces():




#------------------------------------------------------------------------------------------------

def kmeans(data, n_clusters):
    return KMeans(n_clusters=n_clusters).fit(data)

def dbscan(data,eps,min_samples):
    return DBSCAN(eps=eps,min_samples=min_samples).fit(data)

def spectral(data,n_clusters):
    return SpectralClustering(n_clusters=n_clusters).fit(cosine_similarity(data))

def agglo(data,n_clusters):
    return AgglomerativeClustering(n_clusters=n_clusters).fit(data)

