"""
EdgeDetector: a small utility to detect edges in images
It supports three methods: Canny, Sobel and Prewitt
"""

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import sobel, prewitt
from skimage.feature import canny
import numpy as np
import PySimpleGUI as sg

def display_images(img:np.array, img_edges:np.array, cmap:str = "gray") -> None:
    """Displays the image and the edges

    Args:
        img (np.array): The input image
        img_edges (np.array): The image of the edges
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

# Define a layout for our GUI
layout = [
    [sg.Text("Edge detection")],
    [sg.Text("Input image"), sg.FileBrowse(key="input_image")],
    [sg.Text("Method"), sg.Combo(["canny", "sobel", "prewitt"],
     key = "method", default_value="canny", readonly=True)],
    [sg.Text("Colourmap"), sg.Combo(["gray", "viridis", "Greens", "Blues", "Reds"],
     key = "colourmap", default_value="gray", readonly=True)],
    [sg.Button("Detect edges"), sg.Button("Exit")]
]

window = sg.Window("EdgeDetector", layout=layout)

while(True):
    event, values = window.read()
    
    if event=="Exit" or event==sg.WIN_CLOSED:
        window.close()
        break
    
    if event=="Detect edges":
        if values['input_image'] == "":
            sg.popup("Please choose an image first!")
        else:          
            img = imread(values["input_image"])
            img_edges = find_edges(img, values["method"])
            imsave("test_images/DAPI_edges.png", img_edges)
            display_images(img, img_edges, values["colourmap"])        