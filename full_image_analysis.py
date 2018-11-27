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

sklearn_pca = PCA(n_components=30)
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

labelList = newlist

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='right',
            labels=labelList,
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


cluster = df[df["cluster"]==1].sort_values(by= 'image', ascending=True)
print(cluster)
cluster_1 = cluster['image'].tolist()
print(cluster_1)

import matplotlib.image as mpimg

graphics= []
for image in cluster_1:
    graphics.append(io.imread("image_samples_resized/" +image))
    
def show_images(images, cols = 1, titles = None):
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
    
show_images(graphics, cols = 3, titles = cluster_1)
