import os
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import napari
from sklearn.cluster import HDBSCAN
from sklearn.mixture import BayesianGaussianMixture
import shutil

def load_validation_bugs(valid_fp):
    """
    Read all bugs from validation filepath
    Returns a dataframe containing validation centerpoints, number of bugs, and bug types
    """
    valid_y = pd.read_csv(valid_fp+"/validation.csv")
    valid_y['bugs_present']=valid_y['centerpoints'].str.split(";").str[0::4]
    valid_y['n_bugs']=valid_y['centerpoints'].str.split(";").str[0::4].str.len()
    valid_y['z'] = valid_y['centerpoints'].str.split(";").str[1::4]
    valid_y['y'] = valid_y['centerpoints'].str.split(";").str[2::4]
    valid_y['x'] = valid_y['centerpoints'].str.split(";").str[3::4]
    valid_y['centerpoint_coords'] = valid_y.apply(lambda x: [[x['x'][i], x['y'][i], x['z'][i]] for i in range(len(x['x']))], axis=1)
    return valid_y

def threshold(bug):
    """Separate bugs from the background"""
    denoised = ndi.median_filter(bug, size=3) #smoothing technique
    thresholded = denoised > filters.threshold_otsu(denoised)
    return thresholded

def clean(thresholded):
    """Morphology. Grab the areas surrounding each bug and smooth them"""
    width = 20
    remove_holes = morphology.remove_small_holes(
        thresholded, 
        width ** 3
    )
    remove_objects = morphology.remove_small_objects(
        remove_holes, 
        width ** 2,
        connectivity=2
    )
    return remove_objects

def insides(cleaned_bug, technique='local_maxima'):
    """
    Identify insides of bugs, find the points farthest from the edges
    techniques: 'local_maxima', 'h_maxima', 'all_points'
    """

    #euclidian distance from outside to inside to find points furthest inside - think heatmap
    transformed = ndi.distance_transform_edt(cleaned_bug)#, sampling=spacing)
    
    if technique == 'all_points':
        transformed_coords = np.transpose(np.nonzero(transformed)) #grab all inner points
        return transformed_coords
    
    #max grabbing techniques
    if technique == 'local_maxima':
        maxima = morphology.local_maxima(transformed) # grabs lots more max's
    elif technique == 'h_maxima':
        maxima = morphology.h_maxima(transformed, 1) #anything higher than 1 missed small bugs 
    else:
        ValueError("specify a maxima technique")

    #grab all chosen max
    maxima_coords = np.transpose(np.nonzero(maxima))
    return maxima_coords

def higherarchical_clustering(insides, min_cluster_size = 3, min_samples = 2, 
                              cluster_selection_epsilon = 15, metric = 'manhattan'):
    """
    Return centers of bugs by clustering their central points.
    Paramaters same as sklearn HDBSCAN documentation
    """
    clusters = HDBSCAN(
        min_cluster_size=min_cluster_size,             #min number of points in a cluster
        min_samples=min_samples,                  #amount of points that can be considered start of cluster
        cluster_selection_epsilon=cluster_selection_epsilon,   #clusters closer than this get merged
        metric=metric
    ).fit_predict(
        insides   
    )

    center_of_cluster = []
    for clust in np.unique(clusters):
        center_of_cluster += [np.mean(insides[clusters==clust], axis=0)]
    return center_of_cluster

def gaussian_clustering(insides, weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=10000, mean_precision_prior=0.01):
    """
    Return centers of bugs by fitting gaussian distributions to their central points.
    Seems to work better for getting entire bug instead of splitting
    Paramaters same as sklearn BayesianGaussianMixture documentation
    """
    mixtures = BayesianGaussianMixture(
        n_components=min(len(insides), 50),
        covariance_type='tied',
        tol=1e-3,
        reg_covar=0,
        #max_iter=1500,
        init_params="random",  #get random points instead of kmeans (wont cluster well)
        weight_concentration_prior_type=weight_concentration_prior_type,
        weight_concentration_prior=weight_concentration_prior,
        mean_precision_prior=mean_precision_prior,
    ).fit_predict(insides)

    center_of_mix = []
    for mix in np.unique(mixtures):
        center_of_mix += [np.mean(insides[mixtures==mix], axis=0)]
    center_of_mix
    return center_of_mix

def watershed(bug, centers, cleaned_bug, edges=False):
    """
    Segment bug faces or masses using watershed
    """
    #create a matrix like image with zeroes except for centroids
    markers = np.zeros(bug.shape, dtype=np.uint32)
    marker_indices = tuple(np.round(centers).astype(int).T)
    markers[marker_indices] = np.arange(len(centers)) + 1
    #grow the corresponding centroids
    markers_big = morphology.dilation(markers, morphology.ball(5))

    #Segment using Watershed from clustered centers
    if edges:
        segmented = segmentation.watershed(
            edges,
            markers_big,
            mask=cleaned_bug,
        )
        return segmented
    
    segmented = segmentation.watershed(
            cleaned_bug,
            markers_big,
            mask=cleaned_bug,
        )
    return segmented

def output_segments(segmented, bug):
    """Isolate single bugs & recalculate centers"""

    watershed_center_list = []
    single_bug_list = []
    for point in np.unique(segmented):
        #print(point)
        single_bug = np.zeros(bug.shape)
        single_bug[segmented==point] = bug[segmented==point]

        single_bug_list += [single_bug]
        
        watershed_center = np.mean(np.where(single_bug!=0), axis=1).astype(int)
        watershed_center_list += [watershed_center]
    
    return watershed_center_list, single_bug_list


def segment_bugs(mixture_path, clustering_method='gaussian'):
    """
    Pipeline for segmenting bugs, can do 'gaussian' or 'higherarchical'
    """
    #read in bug
    bug = mixture_path
    
    #Edge Detection
    edges = filters.scharr(bug)

    #Thresholding
    thresholded = threshold(bug)

    #Morphology
    cleaned_bug = clean(thresholded)

    #Identify insides of bugs
    technique='local_maxima'
    insides_bug = insides(cleaned_bug, technique)

    #Cluster
    if clustering_method == 'gaussian':
        centers = gaussian_clustering(insides_bug)
    elif clustering_method == 'higherarchical':
        centers = higherarchical_clustering(insides_bug)
    
    #Watershed
    segmented = watershed(bug, centers, cleaned_bug)

    #Isolate single bugs & recalculate centers
    return output_segments(segmented, bug)


