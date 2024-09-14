import numpy as np
import pandas as pd 

from skimage import io # in and out functions from a second image access library
from skimage import measure # Used to define the marching cubes algorithm
from skimage.transform import rotate # Used to rotate volumes
from skimage.transform import rescale # Used to adjust volume dimensions

import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision
from torchvision import transforms

class VolumesDataset(torch.utils.data.Dataset): #torch.utils.data.Dataset
    def __init__(self, data, transform=None, train=False, split_ratio=None):
        bug_type = "BugType"
        self.train= train
        self.split_ratios = split_ratio
        self.train_indices = None
        self.val_indices = None
        self.labels = None
        self.transform = transform
        self.data = data 

        # process the training dataset
        if self.train:
            def balanced(data, ix_train):
                """Balances the given dataset's training data"""
                self.data = data.reset_index()
                df = self.data.copy()
                bug_types = list(data.groupby(bug_type).groups.values()) # indices of each bug type
                num_types = len(bug_types) # number of bug types
                # interpolate bugtypes
                for i in range(num_types):
                    bug_type_ix = np.arange(i, ix_train, num_types) # interpolated indices
                    bug_df = data.loc[bug_types[i]]
                    if bug_df.shape[0] < len(bug_type_ix):
                        bug_type_ix = bug_type_ix[:bug_df.shape[0]]
                    df.loc[bug_type_ix, :] = bug_df.iloc[:len(bug_type_ix)].set_index(bug_type_ix) # assign bug type to interpolated indices
                return df

            # balance the training data
            if self.split_ratios:
                tot = data.shape[0] # number of observations
                train_samples = int(split_ratio[0] * tot) 
                val_samples = int(split_ratio[1] * tot) 
                self.train_indices = np.arange(0, train_samples) # training data
                self.val_indices = np.arange(train_samples, train_samples + val_samples) # validation data
    
                # interpolate the bugs across the dataset to ensure trainingset is balanced
                if split_ratio[0] > 0:
                    self.data = balanced(data, train_samples)
            else:
                print("Please provide the training/validation split ratios.")

            # create the training data labels
            def create_labels(data):
                """Creates the labels for the given dataset"""
                labels = data.BugType.replace(dict(zip(data.BugType.unique(), range(data.BugType.nunique())))).values
                return labels

            self.labels = create_labels(self.data)
            
    def __len__(self):
        if self.train:
            return self.data.shape[0]
        else:
            # test sets will always be 1 file long
            return 1

    def __getitem__(self, idx):
        if self.train:
            img = self.data.FileLoc.iloc[idx]
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]
            return img, torch.tensor(label)
        else:
            img = self.data
            img = self.transform(img)
            return img
        
# read in the data
def read_tif_file(fp):
    '''Read and load the volume'''
    # read file
    im = io.imread(fp)
    return im.astype("float32")

# downsample the 3D volume
def downsample_volume(im, block_size=2):
    im = measure.block_reduce(im, block_size=block_size, func=np.mean)
    return im
    
# resize the volume to uniform dims
def resize_volume(im):
    # set desired dims
    desired_depth = 25
    desired_width = 12
    desired_height = 12
    # get current dims
    curr_depth = im.shape[0]
    curr_width = im.shape[1]
    curr_height = im.shape[2]
    # compute dims factor
    depth_factor = desired_depth / curr_depth
    width_factor = desired_width / curr_width
    height_factor = desired_height / curr_height
    # resize across z-axis
    im = rescale(im, (depth_factor, width_factor, height_factor), order=1, anti_aliasing=True)
    return im


# pre-processing pipeline that resizes and downsamples
def process_vol(fp):
    if type(fp) == str:
        img = read_tif_file(fp) # read in the filepath
    else:
        img = fp.astype("float32") # work with the 3D np array
    img = downsample_volume(img)
    img = resize_volume(img)
    return img

transform_with_unsqueeze = transforms.Compose([
    transforms.Lambda(lambda img: process_vol(img)),
    transforms.Lambda(lambda img: torch.from_numpy(img)),  # Convert to PyTorch tensor
    transforms.Lambda(lambda img: torch.unsqueeze(img, dim=0).repeat(3, 1, 1, 1)) 
])

# make dataset
def make_dataset(df, transforms=transform_with_unsqueeze, train=False, splits=None):
    dataset = VolumesDataset(df, transform=transforms, train=train, split_ratio=splits)
    return dataset

# make dataloader
def load_data(dataset, indices=None, train_batch_size=None):
    if indices is None:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        return torch.utils.data.DataLoader(dataset, sampler=SubsetRandomSampler(indices), num_workers=4, pin_memory=True, batch_size=train_batch_size)