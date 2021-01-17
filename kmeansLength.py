from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from collections import Counter

def show_img_compar(img_1, img_2 ):
    figure, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    figure.tight_layout()
    plt.show()
img = cv.imread("1.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = cv.imread("2.jpg")

img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

clt = KMeans(n_clusters=5)
clt.fit(img.reshape(-1,3))
clt.labels_
clt.cluster_centers_

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    #for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette

clt_1 = clt.fit(img.reshape(-1, 3))
show_img_compar(img, palette_perc(clt_1))
