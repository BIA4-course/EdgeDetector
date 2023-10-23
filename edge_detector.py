"""
EdgeDetector: a small utility to detect edges in images
It supports three methods: Canny, Sobel and Prewitt
"""

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import sobel, prewitt
from skimage.feature import canny
import numpy as np

def display_images(img:np.array, cmap:str = "gray") -> None:
    """Displays the image and the edges

    Args:
        img (np.array): The input image
        cmap (str, optional): The colourmap. Defaults to "gray".
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (16, 8))
    ax[0].imshow(img, cmap=cmap)
    ax[1].imshow(img_edges, cmap=cmap)
    plt.show()

def find_edges(img:np.array, method:str = "canny") -> np.array:
    """Finds edges in an image, using the method chosen by the user

    Args:
        img (np.array): the input image
        method (str, optional): The edge-detection methods. Defaults to "canny".
                                Allowed values are "canny", "prewitt" and "sobel"

    Returns:
        np.array: the image of the edges
    """
    if method == "canny":
        img_edges = canny(img, sigma=7)
    elif method == "sobel":
        img_edges = sobel(img)
    elif method == "prewitt":
        img_edges = prewitt(img)
    else:
        raise ValueError(f"Method {method} is not supported.")
    
    return img_edges
    
# TODO: user needs to select this
img = imread("test_images/DAPI.png")
img_edges = find_edges(img, "canny")

display_images(img)