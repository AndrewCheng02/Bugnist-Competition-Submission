# Custom libraries
import main
import model
import generate_synth_mixtures as sym
import BugNIST_metric as bm

# Datasci tools
import numpy as np
import pandas as pd

# Imaging Tools
from skimage import io

# Generate Custom Data ------------------------------------------------------
def generate_clean_training_data(n = 8000, outpath = None):
    '''
    Generates n clean training data without noise
    '''
    
    pass
    
def generate_synthetic_mixtures(n = 50, outpath = None):
    '''
    Generates n synthetic mixtures
    '''
    
    sym.generate_mixture_set(n)

# Training pipelines --------------------------------------------------------
def train_model_classifier_train(classification_model, path = None, n = 8000, outpath = None):
    '''
    Trains classification model on train data
    
    model: Classification Model class
    n: number of samples
    paths: Training path
    outpath: Where to save the model
    '''
    
    if not path:
        df = pd.DataFrame(train_labels)
        
    else:
        columns = []
        
        for i, v in bug_labels.items():
            for fp in os.listdir(path + i):
                file_path = train_fp + i + '/' + fp
                columns.append({'fp' : file_path, 'type' : v})
        
        df = pd.DataFrame(columns)
    
    df['bug'] = df['fp'].apply(lambda x : io.imread(x))
    
    sample = df.sample(n)
    
    return classification_model.train(sample)


def train_model_classifier_mixtures(classification_model, segmentation_model, path, n = 50, outpath = None):
    '''
    Trains classification model on synthetic mixtures
    
    model: Classification Model class
    path: To csv
    n: number of samples
    outpath: Where to save the model
    '''
    
    path = path if path else synthetic_mixtures_csv_fp
    
    df = pd.read_csv(path)
    
    sample = df.sample(n)
    
    centers, bugs = segmentation_model.segment(sample)
    
    return classification_model.train(bugs, )

def train_model_segmentation_mixtures(segmentation_model, n, outpath = None):
    '''
    Trains segmentation model on synthetic mixtures
    
    model: Segmentaiton Model class
    n: number of samples
    outpath: Where to save the model
    '''
    
    pass

# Validation and Testing Pipelines ----------------------------------------------

def predict_mixture(classification_model, segmentation_model, path):
        
        mixture = io.imread(path)
        
        centers, bugs = segmentation_model.segment(mixture)
        
        predictions = []
        
        for i in range(len(centers)):
            x, y, z = centers[i]
            bug_class = classification_model.classify(bugs[i])
        
            predictions.append(pred + ';' + str(x) + ';' + str(y) + ';' + str(z))
    
        return ';'.join(predictions) 

def run_validation(classification_model, segmentation_model, outpath = None):
    '''
    Gets the validation score of the model
    '''
    
    # Get validation data in data frame
    validation = pd.read_csv(validation_csv_fp)
    
    validation['centerpoints'] = validation['filename'].apply(
        lambda x : predict_mixture(classification_model, segmentation_model, validation_fp + x))
    
    # Save predictions to csv
    validation.set_index('filename').to_csv(out_path if out_path else baseline_csv_fp)

def run_test(classification_model, segmentation_model, outpath = None):
    '''
    Gets the Test output of the model
    '''
    
    # Get test data into data frame
    test = pd.DataFrame([{'filename' : test_file_paths}])
    
    test['centerpoints'] = test['filename'].apply(
        lambda x : predict_mixture(classification_model, segmentation_model, validation_fp + x))
    
    # Save predictions to csv
    test.set_index('filename').to_csv(out_path if out_path else test_csv_fp)
    
def predict(classification_model, segmentation_model, filepath):
    
    '''
    Predicts the data and returns the df
    '''
    
    file_paths = [fp for fp in os.listdir(filepath)]
    
    # Get test data into data frame
    df = pd.DataFrame([{'filename' : file_paths}])
    
    df['centerpoints'] = df['filename'].apply(
        lambda x : predict_mixture(classification_model, segmentation_model, validation_fp + x))
    
    # Return df
    return df.set_index('filename')
    


# Main Function -------------------------------------------------------------------------------
def __main__():
    '''
    Main function to run on DSMLP
    '''
    pass
    
    