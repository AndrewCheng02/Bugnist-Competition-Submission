# ---
# Custom libraries
import main

# Built in libraries
import os
import random

# Data sci tools
import numpy as np
import pandas as pd

# 3D rendering tools
import tifffile
import plotly.graph_objects as go #plotly image library
from scipy.ndimage import center_of_mass

# Skimage
from skimage import io # in and out functions from a second image access library
from skimage import measure # Used to define the marching cubes algorithm
from skimage.transform import resize, rescale, rotate
# -

def plot_surface3(img, ctrs, draw=True):
    """
    Plots surface of multiple bugs and their centers given the image and the bugs' centers
    """
    verts, faces, _, _ = measure.marching_cubes(img)

    # compute the faces and vertices 
    dats = go.Mesh3d(x = verts[:, 0], y = verts[:, 1], z = verts[:, 2],
                                     i = faces[:, 0], j = faces[:, 1], k = faces[:, 2],
                                     opacity = 0.5)
    
    #extract the points and the means
    x_data = dats.x
    y_data = dats.y
    z_data = dats.z
    
    if draw:
        # plot
        fig = go.Figure(data = dats)
        fig.update_layout(scene = dict(xaxis = dict(title='Z'),
                                     yaxis = dict(title='Y'),
                                     zaxis = dict(title='X')),
                          margin = dict(l = 0, r  =0, t = 0, b = 0),
                          title = "Bug Projection")

        # Add a scatter plot for the center point
        new_points = go.Scatter3d(
            x= [i[0] for i in ctrs],  # X coordinate of the new point
            y= [i[1] for i in ctrs],  # Y coordinate of the new point
            z= [i[2] for i in ctrs],  # Z coordinate of the new point
            mode='markers',
            marker=dict(
                size=10,
                color='red',
            ),
            name='Center Of Bug'
        )

        # Add the new points to the figure
        fig.add_trace(new_points)
        fig.show()
    return ctrs 



def embed_matrix(large, small):
    """
    - Large (starts off empty) is the mixture to embed small (bug) to.
    - x_start, y_start, z_start: random indices in mixture matrix, 
    bounded to ensure small (bug) can fully fit. Where small is to be 
    embedded into 
    - xm, ym, zm: center indices of mixture. The while loop starts with
    a small pad around these positions, biasing bugs to be inserted
    towards center of the image, if they fit. As loop iterates, padding
    expands.
    - nx: number of non-zero pixels where small (bug) is to be inserted
    into. This represents if another bug already occupies the space in 
    the mixture matrix that a new bug is to be inserted into. 0.0001 can
    be modified depending on prefered tolerance for overlapping bugs.
    - Additional things: rotates the smaller matrix
    - Return: Modified large matrix, a flag indicating whether embedding was 
    successful, position of the embedded bug (if successful)
    """
    ctr = tuple(np.round(i, 2) for i in center_of_mass(small))  # Center of mass of small matrix
    new_ctr = np.nan
    small = rotate(small, random.randint(0, 360), resize=False)  # rotate image to random position

    found = False
    n_attempts = 0  # Number of embedding attempts
    pad = 1

    xs, ys, zs = small.shape
    xl, yl, zl = large.shape
    xm, ym, zm = tuple(i // 2 for i in large.shape)

    xmax = xl - xs
    ymax = yl - ys
    zmax = zl - zs

    while not found:  # Haven't found a spot to insert the small matrix
        
        pad += 1.5
        sigma = pad / 3  # Standard deviation for Gaussian distribution
        
        x_start = int(np.clip(np.random.normal(xm, sigma), 0, xmax))
        y_start = int(np.clip(np.random.normal(ym, sigma), 0, ymax))
        z_start = int(np.clip(np.random.normal(zm, sigma), 0, zmax))

        nz = (large[
            x_start:x_start + small.shape[0],
            y_start:y_start + small.shape[1],
            z_start:z_start + small.shape[2]
        ] != 0).sum() / np.prod(small.shape)

        if nz <= 0.1:
            large[x_start:x_start + small.shape[0],
                  y_start:y_start + small.shape[1],
                  z_start:z_start + small.shape[2]] = small
            found = True
            new_ctr = (ctr[0] + x_start, ctr[1] + y_start, ctr[2] + z_start)
        
        if n_attempts == 100:
            # Max limit reached
            break
        else:
            n_attempts += 1
            
    new_ctr = np.round(new_ctr, 2)
    return large, found, new_ctr



def isolate_bug(bug):
    """
    Isolate the bug by removing as much white space as possible
    """
    x, y, z = np.where(bug != 0)
    isolated_bug = bug[
        x.min():x.max()+1, y.min():y.max()+1, z.min():z.max()+1
    ]
    return isolated_bug


def generate_mixtures(df):
    """
    Randomly generates a number (n) from 4-16, then randomly generates a bug to embed into blank matrix
    until n number of bugs are embedded or 50 different bugs have been tried
    """
    mixture = np.zeros((128, 92, 92))  # initialize blank matrix with all zeros
    num_files = random.randint(4, 16)  # Generate a random number between 4 and 15
#     print("Number of files: " + str(num_files))
    
    count = 0
    ctrs = []
    filepaths = []
    n_attempts = 0
    while count != num_files:
        random_filepath = df['fp'].sample(n=1).iloc[0] # sample for a random bug
        try: 
            isolated_bug = isolate_bug(tifffile.imread(random_filepath))
        except:
            continue
        mixture, found, ctr = embed_matrix(mixture, isolated_bug)
        if found:
            ctrs.append(ctr)
            filepaths.append(random_filepath)
            count += 1
        n_attempts += 1
        if n_attempts == 50:
            break
    return mixture, filepaths, ctrs


# +
def generate_mixture_set(n):
    """Generates n number of mixtures and puts them into a foler"""
    filenames = []
    centerpoints = []
    file_directory = synthetic_mixtures_fp
    for iteration in range(n):
        # generate bug mixture
        mixture = generate_mixtures(data)

        # identify training bugs used
        the_bugs = [i.split('/')[3].lower() for i in mixture[1]]

        # get the centers
        the_centers = []
        for i in mixture[2]:
            sub = []
            for j in list(i):
                sub.append(str(j))
            the_centers.append(';'.join(sub))

        the_centerpoint = ';'.join([k + ';' + m for k,m in zip(the_bugs, the_centers)])
        centerpoints.append(the_centerpoint)
        filename = 'synth_mixture_' + str(iteration+1) + '.tif'
        filenames.append(filename)
        location = file_directory+filename
        tifffile.imsave(location, mixture[0])
        
        syn_mixtures = pd.DataFrame({'filename':filenames, 'centerpoints':centerpoints})
        syn_mixtures.to_csv(synthetic_mixtures_fp + 'synthetic_mixtures.csv')
    
    
