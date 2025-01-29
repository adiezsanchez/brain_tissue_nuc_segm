from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure, segmentation
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

def fix_overlapping_labels(lbl):
    """If the most common neighboring label is 0 (background), the original label r.label is preserved.
    Otherwise, the region is reassigned to the neighboring outer_label""" 
    y = np.zeros_like(lbl) 
    for r in measure.regionprops(lbl):  
        outer = list(lbl[segmentation.find_boundaries(lbl==r.label, mode=  "outer")])  
        outer_label = max(outer,key=outer.count)  
        y[lbl==r.label] = r.label if outer_label==0 else outer_label 
    return y 

def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y

def ignore_xy_border_labels(labels, buffer_size=0, border_val=-1, out=None):
    """Set objects connected to the x and y edges of the label image to -1 to be ignored during Stardist training.

    Parameters
    ----------
    labels : (Z, Y, X) array of int or bool
        Imaging data labels.
    buffer_size : int, optional
        The width of the border examined. Default is 0, meaning only objects touching
        the very edge are removed.
    border_val : float or int, optional
        The value to assign to labels touching the border. Default is -1.
    out : ndarray, optional
        Array of the same shape as `labels`, into which the output is placed.
        By default, a new array is created.

    Returns
    -------
    out : (Z, Y, X) array
        Imaging data labels with cleared (set to -1) x and y edge-connected objects.
    """

    # Ensure labels are int16 to allow negative values
    labels = labels.astype(np.int16)

    if any(buffer_size >= s for s in labels.shape[1:]):
        raise ValueError("buffer size may not be greater than labels size in x or y dimensions")

    if out is None:
        out = labels.copy()

    # Create borders in x and y dimensions only
    borders = np.zeros_like(out, dtype=bool)
    ext = buffer_size + 1

    # Apply border mask to only x and y edges
    borders[:, :ext, :] = True  # Front y-edge
    borders[:, -ext:, :] = True  # Back y-edge
    borders[:, :, :ext] = True  # Left x-edge
    borders[:, :, -ext:] = True  # Right x-edge

    # Re-label the image
    labeled_img, number = measure.label(out, background=0, return_num=True)

    # Identify objects connected to x or y borders
    borders_indices = np.unique(labeled_img[borders])
    #print("Border-connected labels:", borders_indices)  # Debugging step

    # Create mask for pixels belonging to border-connected labels
    label_mask = np.isin(labeled_img, borders_indices)

    # Only modify label pixels, keep background (0) unchanged
    out[(label_mask) & (out > 0)] = border_val  # Set only labels to `border_val`, keep background as 0

    return out