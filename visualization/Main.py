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
from sklearn.preprocessing import StandardScaler

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
    data = np.load('data_lbp_feature.npy')
    target = np.load('data_target.npy')
    return data, target
def init_hog_faces():
    data = np.load('data_hog_feature.npy')
    target = np.load('data_target.npy')
    return data, target

#------------------------------------------------------------------------------------------------

def tech_kmeans(data, n_clusters):
    return KMeans(n_clusters=n_clusters).fit(data)
def tech_dbscan(data,eps,min_samples):
    return DBSCAN(eps=eps,min_samples=min_samples,algorithm='kd_tree').fit(StandardScaler().fit_transform(data))
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

def load_dataset():
    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    data_lbp = []
    data_hog = []
    target = faces.target
    for image in faces.images:
        feature_lbp = local_binary_pattern(image, P=24, R=4)
        _, feature_hog = hog(image, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1, 1), visualise=True)
        data_lbp.append(feature_lbp.flatten())
        data_hog.append(feature_hog.flatten())
    np.save(file='data_target.npy', arr=target)
    np.save(file='data_lbp_feature.npy', arr=data_lbp)
    np.save(file='data_hog_feature.npy', arr=data_hog)
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
    t_kmeans = time() - t_origin
    f_kmeans = compare(kmeans.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using KMeans.",data_digits,kmeans.labels_)

    t_origin = time()
    dbscan = tech_dbscan(data=data_digits, eps=5.541, min_samples=81)
    t_dbscan = time() - t_origin
    f_dbscan = compare(dbscan.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using DBSCAN.",data_digits,dbscan.labels_)

    t_origin = time()
    spectral = tech_spectral(data=data_digits, n_clusters=10)
    t_spectral = time() - t_origin
    f_spectral = compare(spectral.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using Spectral Clustering.",data_digits,spectral.labels_)

    t_origin = time()
    agglo = tech_agglo(data=data_digits, n_clusters=10)
    t_agglo = time() - t_origin
    f_agglo = compare(agglo.labels_, target_digits)
    figure("Hw02 - Clustering Handwritten digits using Agglomerative.",data_digits,agglo.labels_)

    print("\t\tKMeans\t\tDBSCAN\t\tSpectral\tAgglomerative")
    print("Time:\t%f\t%f\t%f\t%f" %(t_kmeans, t_dbscan, t_spectral, t_agglo))
    print("%%Fit:\t%f\t%f\t%f\t%f" %(f_kmeans, f_dbscan, f_spectral, f_agglo))
def part_c():
    data_faces, target_faces = init_lbp_faces()
    figure("Hw03 - Clustering Human Faces target (Feature-LBP).",data_faces,target_faces)

    t_origin = time()
    kmeans = tech_kmeans(data=data_faces, n_clusters=10)
    t_kmeans = time() - t_origin
    f_kmeans = compare(kmeans.labels_, target_faces)
    figure("Hw03 - Clustering Human Faces using KMeans.",data_faces,kmeans.labels_)

    t_origin = time()
    dbscan = tech_dbscan(data=data_faces, eps=41, min_samples=1)
    t_dbscan = time() - t_origin
    f_dbscan = compare(dbscan.labels_, target_faces)
    figure("Hw03 - Clustering Human Faces using DBSCAN.",data_faces,dbscan.labels_)

    t_origin = time()
    spectral = tech_spectral(data=data_faces, n_clusters=10)
    t_spectral = time() - t_origin
    f_spectral = compare(spectral.labels_, target_faces)
    figure("Hw03 - Clustering Human Faces using Spectral Clustering.",data_faces,spectral.labels_)

    t_origin = time()
    agglo = tech_agglo(data=data_faces, n_clusters=10)
    t_agglo = time() - t_origin
    f_agglo = compare(agglo.labels_, target_faces)
    figure("Hw03 - Clustering Human Faces using Agglomerative.",data_faces,agglo.labels_)

    print("Datasets which extracted by Local Binary Pattern method: ")

    print("\t\tKMeans\t\tDBSCAN\t\tSpectral\tAgglomerative")
    print("Time:\t%f\t%f\t%f\t%f" %(t_kmeans, t_dbscan, t_spectral, t_agglo))
    print("%%Fit:\t%f\t%f\t%f\t%f" %(f_kmeans, f_dbscan, f_spectral, f_agglo))
def part_d():
    data_faces, target_faces = init_hog_faces()
    figure("Hw04 - Clustering Human Faces target (Feature-HOG).",data_faces,target_faces)

    t_origin = time()
    kmeans = tech_kmeans(data=data_faces, n_clusters=10)
    t_kmeans = time() - t_origin
    f_kmeans = compare(kmeans.labels_, target_faces)
    figure("Hw04 - Clustering Human Faces using KMeans.",data_faces,kmeans.labels_)

    t_origin = time()
    dbscan = tech_dbscan(data=data_faces, eps=19, min_samples=1)
    t_dbscan = time() - t_origin
    f_dbscan = compare(dbscan.labels_, target_faces)
    figure("Hw04 - Clustering Human Faces using DBSCAN.",data_faces,dbscan.labels_)

    t_origin = time()
    spectral = tech_spectral(data=data_faces, n_clusters=10)
    t_spectral = time() - t_origin
    f_spectral = compare(spectral.labels_, target_faces)
    figure("Hw04 - Clustering Human Faces using Spectral Clustering.",data_faces,spectral.labels_)

    t_origin = time()
    agglo = tech_agglo(data=data_faces, n_clusters=10)
    t_agglo = time() - t_origin
    f_agglo = compare(agglo.labels_, target_faces)
    figure("Hw04 - Clustering Human Faces using Agglomerative.",data_faces,agglo.labels_)

    print("Datasets which extracted by Histogram of Gradient method: ")

    print("\t\tKMeans\t\tDBSCAN\t\tSpectral\tAgglomerative")
    print("Time:\t%f\t%f\t%f\t%f" %(t_kmeans, t_dbscan, t_spectral, t_agglo))
    print("%%Fit:\t%f\t%f\t%f\t%f" %(f_kmeans, f_dbscan, f_spectral, f_agglo))

#------------------------------------------------------------------------------------------------

def main():
    #part_a()
    #part_b()
    #part_c()
    part_d()
    return 0

main()
