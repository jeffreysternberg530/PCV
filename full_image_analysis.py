import matplotlib.pyplot as plt

import os
import numpy as np

from skimage import io

from sklearn.decomposition import PCA

import glob


data = os.listdir("image_samples_resized")

newlist = []
for photos in data:
    if photos.endswith(".jpg"):
        newlist.append(photos)
        
img = []
for d in newlist:
    img.append(io.imread("image_samples_resized/" +d).mean(axis=2).flatten())
    
img = np.array(img)

m,n = 209,247

sklearn_pca = PCA(n_components=30, random_state=4)
Y_sklearn = sklearn_pca.fit_transform(img)

import pandas as pd
df = pd.DataFrame()

df["image"] = newlist

from scipy.cluster.hierarchy import dendrogram, linkage  
from scipy.cluster.hierarchy import fcluster

linked = linkage(Y_sklearn, 'ward')

max_d = 32000
clusters = fcluster(linked, max_d, criterion='distance')
    
df['cluster'] = clusters

labelList = ['' for i in range(0,138)]

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='right',
            labels = labelList,
            distance_sort='descending',
            show_leaf_counts=True,
            show_contracted=True,
            color_threshold=32000)
plt.xlabel('Distance', fontsize=24)
plt.xticks(fontsize = 18)
plt.tight_layout()  # fixes margins

plt.axvline(x=32000) #plot vertical line

plt.show()

print(df.sort_values(by= ['cluster','image'], ascending=True))

def cluster_lister(df, column):
    
    cluster_list = []
    k = df[column].max()
    x = df[column].min()
    
    k=k+1
    
    for i in range(x,k):
    
        cluster_i= df[df[column]==i].sort_values(by= 'image', ascending=True)
        cluster_i_list = cluster_i['image'].tolist()
        cluster_list.append(cluster_i_list)
        
    return cluster_list
    
def image_lister(cluster_list):
    
    image_list = []
    k = len(cluster_list)
    
    for i in range(k):
        img_i = []
        for image in cluster_list[i]:
            img_i.append(io.imread("image_samples_resized/" +image))
            
        image_list.append(img_i)
            
    return image_list
    
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
def image_cluster_viewer(image_list, cluster_list):
    
    k = len(image_list)
    
    for i in range(k):
        show_images(image_list[i], cols = 3, titles = cluster_list[i])

        
cluster_list = cluster_lister(df, 'cluster')
image_list = image_lister(cluster_list)
image_cluster_viewer(image_list, cluster_list)



