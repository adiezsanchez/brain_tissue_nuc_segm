from cellpose import models
from pathlib import Path
import numpy as np
from skimage import exposure, filters, measure
from scipy.ndimage import binary_erosion
import pyclesperanto_prototype as cle

cle.select_device("RTX")

# Define the Cellpose model that will be used
model = models.Cellpose(gpu=True, model_type="nuclei")

def segment_nuclei_2d (nuclei_input, gaussian_sigma = 0, cellpose_nuclei_diameter = None):

    if len(nuclei_input.shape) == 3:
        # Perform maximum intensity projection (MIP) from the stack
        nuclei_mip = np.max(nuclei_input, axis=0)

    elif len(nuclei_input.shape) == 2:
        # Input is already a maximum intensity projection (MIP)
        nuclei_mip = nuclei_input

    # Might need to perform a Gaussian-blur before
    post_gaussian_img = filters.gaussian(
        nuclei_mip, sigma=gaussian_sigma
    )

    # Apply Contrast Stretching to improve Cellpose detection of overly bright nuclei
    p2, p98 = np.percentile(post_gaussian_img, (2, 98))
    img_rescale = exposure.rescale_intensity(
        post_gaussian_img, in_range=(p2, p98)
    )

    # Predict nuclei nuclei_masks using cellpose
    nuclei_labels, flows, styles, diams = model.eval(
        img_rescale,
        diameter=cellpose_nuclei_diameter,
        channels=[0, 0],
        net_avg=False,
    )
    if len(nuclei_input.shape) == 3:
        return nuclei_mip, nuclei_labels
    elif len(nuclei_input.shape) == 2:
        return nuclei_labels

def segment_marker_positive_nuclei (nuclei_labels, marker_input, marker_channel_threshold, erosion_factor):

    if len(marker_input.shape) == 3:
        # Perform maximum intensity projection from the stack
        marker_mip = np.max(marker_input, axis=0)

    elif len(marker_input.shape) == 2:
        # Input is already a maximum intensity projection (MIP)
        marker_mip = marker_input

    # Convert nuclei_masks to boolean mask
    nuclei_masks_bool = nuclei_labels.astype(bool)

    # Find nuclei that intersect with the marker signal
    nuclei_and_marker = nuclei_masks_bool & (marker_mip > marker_channel_threshold)

    # Erode the result to remove small objects
    structuring_element = np.ones((erosion_factor, erosion_factor), dtype=bool)
    eroded_nuclei_and_marker = binary_erosion(nuclei_and_marker, structure=structuring_element)

    # Label the eroded nuclei and marker mask
    labeled_nuclei, num_labels = measure.label(nuclei_labels, return_num=True)

    # Use NumPy's advanced indexing to identify labels that intersect with the eroded marker mask
    intersecting_labels = np.unique(labeled_nuclei[eroded_nuclei_and_marker])
    intersecting_labels = intersecting_labels[intersecting_labels != 0]  # Remove background label

    # Create an empty array for the final labeled nuclei
    processed_region_labels = np.zeros_like(labeled_nuclei, dtype=int)

    # Recover the full extent of nuclei that intersect with the marker mask
    for idx, label in enumerate(intersecting_labels):
        # Recover the entire region of the original nuclei_mask that has this label
        processed_region_labels[labeled_nuclei == label] = label

    return nuclei_and_marker, eroded_nuclei_and_marker, marker_mip, processed_region_labels

def check_filenames(images, rois):

    # Extract the base filenames without extensions using Path.stem
    images_base = [Path(file).stem for file in images]
    rois_base = [Path(file).stem for file in rois]

    # Check for missing files in images list
    missing_in_images = [file for file in rois_base if file not in images_base]
    if missing_in_images:
        for file in missing_in_images:
            print(f"Missing in images list: {file}")
    else:
        print("No files missing in images list.")

    # Check for missing files in rois list
    missing_in_rois = [file for file in images_base if file not in rois_base]
    if missing_in_rois:
        for file in missing_in_rois:
            print(f"Missing in rois list: {file}")
    else:
        print("No files missing in rois list.")

def simulate_cytoplasm(nuclei_labels, dilation_radius = 2, erosion_radius = 0):

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cyto_nuclei_labels = cle.pull(cyto_nuclei_labels)
    cytoplasm = cyto_nuclei_labels

    # Create a copy of dilated_nuclei to modify
    # cytoplasm = cyto_nuclei_labels.copy()

    # Get unique labels (excluding 0 which is background)
    unique_labels = np.unique(nuclei_labels)
    unique_labels = unique_labels[unique_labels != 0]

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Iterate over each label and remove the corresponding pixels from dilated_nuclei
    for label in unique_labels:
        # Create a mask for the current label in filtered_nuclei
        mask = (nuclei_labels == label)
        # Set corresponding pixels in resulting_nuclei to zero
        cytoplasm[mask] = 0

    return cytoplasm