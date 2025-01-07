from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stardist import random_label_cmap

np.random.seed(42)
lbl_cmap = random_label_cmap()

def read_image(image, slicing_factor):
    """Read .tif files and apply a slicing factor, returns the compressed image"""
    # Read path storing raw image and extract filename
    file_path = Path(image)

    # Read .tif file
    img = imread(file_path)
    img = img[:, ::slicing_factor, ::slicing_factor]

    return img

def process_images(image_list, slicing_factor):
    return list(map(lambda img: read_image(img, slicing_factor), image_list))

def plot_img_label(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None, **kwargs):
    if z is None:
        z = img.shape[0] // 2    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()