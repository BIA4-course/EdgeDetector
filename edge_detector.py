"""
EdgeDetector: a small utility to detect edges in images
It supports three methods: Canny, Sobel and Prewitt
"""

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import sobel, prewitt
from skimage.feature import canny
import numpy as np
import argparse
import os

# We will call our software as
# python edge_detector.py --input inputfile.png [--output outputfile.png] [--method canny] [--cmap gray]
parser = argparse.ArgumentParser(description="Finds edges in the provided image")
parser.add_argument("-i", "--input", help="The input image file name", required=True)
parser.add_argument("-o", "--output", help="The output image file name", required=False)
parser.add_argument("-m", "--method", help="The edge detecting method, default is canny", 
                    required=False, default="canny", choices=["canny", "sobel", "prewitt"])
parser.add_argument("-c", "--cmap", help="The colourmap to display the image, default is gray", 
                    default="gray")

args = parser.parse_args()
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
    
if os.path.exists(args.input):
    img = imread(args.input)
else:
    raise FileExistsError(f"Input file {args.input} does not exist!")

img_edges = find_edges(img, args.method)
# If the user does not provide an output file name generate it automatically
if args.output is None:
    # input.png -> input_edges.png
    input_name_split = args.input.split(".")
    output_filename = f"{input_name_split[0]}_edges.{input_name_split[1]}"
else:
    output_filename = args.output
    
imsave(output_filename, img_edges)
display_images(img, img_edges, cmap=args.cmap)