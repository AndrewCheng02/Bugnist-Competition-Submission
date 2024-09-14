import main

import segment_mixture
import VolumesDataset
import CNNModel
import torch

import numpy as np
import pandas as pd
import os

# Imaging Tools
from skimage import io

def predict_bugs(bug_np, newmodel):
    # make dataset
    bug_data = VolumesDataset.make_dataset(bug_np)
    # make dataloader
    bug_loader = VolumesDataset.load_data(bug_data)    
    # make preds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    newmodel.to(device)
    newmodel.eval()
    for img in bug_loader:
        with torch.no_grad():
            img = img.to(device)
            out = newmodel(img)
            pred = torch.max(out.data, 1)[1]
    return pred[0].item()


def predict(data_path, pkl_path):
    """Predict bugnist data.

    Parameters
    ----------
    data_path : str
        Path to data directory,e.g.
        "bugnist2024fgvc/BugNIST_DATA/validation" or
        "bugnist2024fgvc/BugNIST_DATA/test"
    pkl_path : str
        Path to additional required data. Here, it's
        the weights of the torch model, "model.pkl"

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """

    classification_model = CNNModel.load_in_model(pkl_path)
    
    def predict_mixture(file_path):
        
        mixture = io.imread(data_path + file_path)
        
        centers, bugs = segment_mixture.segment_bugs(mixture)
        
        predictions = []
        
        for i in range(len(centers)):
            x, y, z = centers[i]
            bug_class = predict_bugs(np.array(bugs[i]), classification_model)
            pred = main.int_to_bug[bug_class]
        
            predictions.append(pred + ';' + str(x) + ';' + str(y) + ';' + str(z))
    
        return ';'.join(predictions) 
    
    # Get test data into data frame
    df = pd.DataFrame({'filename' : [fp for fp in os.listdir(data_path)]})
    df['centerpoints'] = df['filename'].apply(predict_mixture)
    
    
    return df

