# Custom libraries
import main
import segment_mixture as sm

# Built in libraries
import os
import random

# Data sci tools
import numpy as np
import pandas as pd

# Abstract Classes ------------------------------------
class Model:
    
    def __init__(self):
        pass
    
    def train(self, dataframe, n):
        '''
        Dataframe contains the bug object (imread obj) and it's labels
        '''
        pass
    

class Classification_Model(Model):
    
    def classify(self, data):
        pass

class Segmentation_Model(Model):
        
    def segment(self, mixture):
        '''
        mixture is a io read tiffile
        '''
        pass
    
    def predict_num_of_bug(self, mixture):
        '''
        mixture is a io read tiffile
        '''
        
        return len(segment(mixture))
    
    def score(self, mixture, label):
        
        # Score Predicting Correct num of bugs
        print('')
        
        # Score for Correct Classification
        print('')
        
        # Score for Correct Center Points (Distance)
        print('')
        
# Classifiers -----------------------------------------

    
# Segmentation Model ----------------------------------

class Higherarchical_Watershed(Segmentation_Model):
    '''
    Higherarchical Clustering with Watershed Seperation
    '''
        
    def segment(self, mixture):
        '''
        mixture is a io read tiffile
        '''
        
        return sm.segment_bugs(mixture, 'higherarchical')

class Gaussian_Watershed(Segmentation_Model):
    '''
    Gaussian Clustering with Watershed Seperation
    '''
        
    def segment(self, mixture):
        '''
        mixture is a io read tiffile
        '''

        return sm.segment_bugs(mixture, 'gaussian')

# Helper Classes -----------------------------------------------------------------------
