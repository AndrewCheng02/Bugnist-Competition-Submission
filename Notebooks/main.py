# Helper Methods and Global Variables
import os
import random

# Root file path
root = os.path.abspath('../') + '/'


# Data File Paths
data_fp = root + 'BugNIST_DATA/'

train_fp = data_fp + 'train/'
validation_fp = data_fp + 'validation/'
test_fp = data_fp + 'test/'
custom_train_fp = data_fp + 'custom_train/'
synthetic_mixtures_fp = data_fp + 'synthetic_mixture/'

# Train Labels File Paths
bug_labels = {'AC' : 'Brown Cricket', 'BC' : 'Black Cricket', 'BF' : 'Blow fly', 
              'BL' : 'Buffalo Beetle Larva' , 'BP' : 'Blow Fly Pupa', 
              'CF' : 'Curly-wing Fly', 'GH' : 'Grasshopper',
              'MA' : 'Maggot', 'ML' : 'Mealworm', 'PP' : 'Green Bottle Fly Pupa' , 
              'SL' : 'Soldier Fly Larva', 'WO' : 'Woodlice'}

# Integer to labels
int_to_bug = {0: 'ac', 1: 'bc', 2: 'bf', 3: 'bl', 4: 'bp', 5: 'cf', 
              6: 'gh', 7: 'ma', 8: 'ml', 9: 'pp', 10: 'sl', 11: 'wo'}

train_labels = []
train_file_paths = []

for i, v in bug_labels.items():
    for fp in os.listdir(train_fp + i):
        
        file_path = train_fp + i + '/' + fp
        
        train_file_paths.append(file_path)
        train_labels.append({'fp' : file_path, 'type' : v})

# Validation File Paths
validation_csv_fp = validation_fp + 'validation.csv'

# Test File Paths
test_file_paths = [fp for fp in os.listdir(test_fp)]

# Synthetic File Paths
synthetic_mixtures_file_paths = [fp for fp in os.listdir(synthetic_mixtures_fp)]
synthetic_mixtures_csv_fp = synthetic_mixtures_fp + 'synthetic_mixtures.csv'

# Models File Paths
models_fp = root + 'Models/'

# Results File Paths
results_fp = root + 'Results/'
baseline_csv_fp = results_fp + 'baseline.csv'
test_csv_fp = results_fp + 'test.csv'


# Helper Methods ---------------------------------------------------------------------------------
def parse_mixture_label(label):
    
    arr = label.split(';')
    
    dic = {} # Coords that point to a bug class
    
    for i in range(0,len(arr),4):
        
        x, y, z, bug_class = arr[i:i+4]  
        dic[(float(x), float(y), float(z))] = bug_class
        
    return dic
    
    
    
    