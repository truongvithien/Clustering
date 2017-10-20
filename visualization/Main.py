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
from sklearn.metrics    import pairwise, adjusted_mutual_info_score
from time import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

from sklearn.datasets   import samples_generator, load_digits, fetch_lfw_people
from sklearn.decomposition import PCA

from skimage.feature    import local_binary_pattern, hog
from sklearn.cluster    import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering

#------------------------------------------------------------------------------------------------

def init_blobs(n_samples, n_features):
    n_samples = 1000
    n_features = 2
    data, _ = samples_generator.make_blobs(n_samples=n_samples, n_features=n_features)
    return data

def init_digits():
    digits = load_digits()
    data = digits.data
    target = digits.target
    return data, target

def init_lbp_faces():
    data = np.load('lbp_data_feature.npy')
    target = np.load('lbp_data_target.npy')
    return data, target

def init_hog_face():
    data = np.load('hog_data_feature.npy')
    target = np.load('hog_data_target.npy')
    return data, target

#------------------------------------------------------------------------------------------------

def tech_kmeans(data, n_clusters):
    return KMeans(n_clusters=n_clusters).fit(data)

def tech_dbscan(data,eps,min_samples):
    return DBSCAN(eps=eps,min_samples=min_samples).fit(data)

def tech_spectral(data,n_clusters):
    return SpectralClustering(n_clusters=n_clusters).fit(pairwise.cosine_similarity(data))

def tech_agglo(data,n_clusters):
    return AgglomerativeClustering(n_clusters=n_clusters).fit(data)

#------------------------------------------------------------------------------------------------

def compare(labels, targets):
    return adjusted_mutual_info_score(targets, labels)

def figure(title, data, labels):
    data = reduced_data = PCA(n_components=2).fit_transform(data)
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='.', s=69, linewidths=3, zorder=10)
    plt.show()

#------------------------------------------------------------------------------------------------

def part_a():
    data_blobs = init_blobs(n_samples=20000, n_features=2)
    t_origin = time()
    kmeans = tech_kmeans(data=data_blobs, n_clusters=3)
    t_kmeans = t_origin - time()
    print("Time: %d" %t_kmeans)
    figure("Hw01 - Clustering generated blobs using KMeans.",data_blobs,kmeans.labels_)

def part_b():
    data_digits, target_digits = init_digits()
    figure("Hw02 - Clustering Handwritten digits target.",data_digits,target_digits)

    t_origin = time()
    kmeans = tech_kmeans(data=data_digits, n_clusters=10)
    t_kmeans = t_origin - time()
    f_kmeans = compare(kmeans.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using KMeans.",data_digits,kmeans.labels_)

    t_origin = time()
    dbscan = tech_dbscan(data=data_digits, eps=1, min_samples=1)
    t_dbscan = t_origin - time()
    f_dbscan = compare(dbscan.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using DBSCAN.",data_digits,dbscan.labels_)

    t_origin = time()
    spectral = tech_spectral(data=data_digits, n_clusters=10)
    t_spectral = t_origin - time()
    f_spectral = compare(spectral.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using Spectral Clustering.",data_digits,spectral.labels_)

    t_origin = time()
    agglo = tech_agglo(data=data_digits, n_clusters=10)
    t_agglo = t_origin - time()
    f_agglo = compare(agglo.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using Agglomerative.",data_digits,agglo.labels_)


    print("\tKMeans\tDBSCAN\tSpectral\tAgglomerative")
    print("Time:\t%f\t%f\t%f\t%f" %(t_kmeans, t_dbscan, t_spectral, t_agglo))
    print("Fit:\t%f\t%f\t%f\t%f" %(f_kmeans, f_dbscan, f_spectral, f_agglo))



#------------------------------------------------------------------------------------------------

def main():
    part_b()
    return 0

main()
