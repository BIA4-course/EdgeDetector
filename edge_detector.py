"""
EdgeDetector: a small utility to detect edges in images
It supports three methods: Canny, Sobel and Prewitt
"""

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import sobel, prewitt
from skimage.feature import canny

# TODO: user needs to select this
img = imread("test_images/DAPI.png")
img_edges = canny(img, sigma=7)

# Display image and results
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (16, 8))
ax[0].imshow(img, cmap="gray")
ax[1].imshow(img_edges, cmap="gray")
plt.show()
