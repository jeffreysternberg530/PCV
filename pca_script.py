import pickle
from scipy.cluster.vq import *
import glob

from PCV.clustering import hcluster
from PCV.tools import pca
from PCV.tools import imtools

# get list of images

imlist = glob.glob("image_sample_copy/animal_images/*.jpeg")
# imlist = imtools.get_imlist('data/image_sample/animal_images')

imlist

from PIL import Image
from numpy import *
import numpy as np
from pylab import *

im = np.array(Image.open(imlist[0])) # open one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

arraylist = [array(Image.open(im)).astype(float) for im in imlist]

arraylist

shapelist = [array.data.shape for array in arraylist]

shapelist

shapelistmean = (mean([shape[0] for shape in shapelist]),mean([shape[1] for shape in shapelist]))

shapelistmean

import os
#size = 159, 225
size = 201, 225
for infile in imlist:
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).resize(size)
    im.save(file + "_resized.jpg", "JPEG")
    
imlist_resized = glob.glob("image_sample_copy/animal_images/*.jpg")

imlist_resized

# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist_resized])

shapes = [im.data.shape for im in immatrix]

shapes

# perform PCA
V,S,immean = pca.pca(immatrix)

S

# save mean and principal components
f = open('sample_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()